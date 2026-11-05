use baux::semantics::optimizer::portfolio::{Allocation, RegionalLedger};

#[test]
fn independent_completion() {
    let mut ledger = RegionalLedger::new(&[0.0, 1.0], Allocation::Balanced).unwrap();
    let first_slot = ledger.allocate(1, &[true, true]).unwrap()[0].region;
    let first = ledger.issue(first_slot).unwrap();
    let second_slot = ledger.allocate(1, &[true, true]).unwrap()[0].region;
    let second = ledger.issue(second_slot).unwrap();
    assert!(ledger.allocate(1, &[true, true]).is_err());
    assert!(ledger.complete(first, &[f64::NAN], 0.0).is_err());
    ledger.complete(second, &[3.0], 3.0).unwrap();
    assert!(ledger.complete(second, &[4.0], 4.0).is_err());
    ledger.complete(first, &[2.0], 2.0).unwrap();
    assert_eq!(ledger.trials(), vec![1, 1]);
    assert_eq!(ledger.best_region(), (1, 3.0));
    ledger.restart(0, 4.0).unwrap();
    assert_eq!(ledger.generation(0).unwrap(), 1);
    assert!(ledger.validate(first).is_err());
    let foreign = RegionalLedger::new(&[0.0], Allocation::Balanced).unwrap();
    assert!(foreign.validate(second).is_err());
}

#[test]
fn retired_completion() {
    let mut ledger = RegionalLedger::new(&[0.0], Allocation::Balanced).unwrap();
    let ticket = ledger.issue(0).unwrap();
    ledger.retire(ticket).unwrap();
    assert!(ledger.validate(ticket).is_err());
    assert_eq!(ledger.trials(), vec![0]);
    let fresh = ledger.issue(0).unwrap();
    ledger.complete(fresh, &[1.0], 1.0).unwrap();
    assert_eq!(ledger.trials(), vec![1]);
}

#[test]
fn resident_completion() {
    use baux::{
        TRLengthConfig,
        resident_metal::{ParamBlock, RegionalSearch, SearchState},
        trials::RoundConfig,
    };
    let state = |value| {
        SearchState::new(
            &[0x3e80; 17],
            value,
            0.0,
            vec![ParamBlock::new(1, 0, 17, 0.25, 1.0).unwrap()],
            3,
            1,
            TRLengthConfig::new(0.1, 0.001, 0.8),
        )
        .unwrap()
    };
    let mut search =
        RegionalSearch::new(vec![state(0.0), state(1.0)], Allocation::Balanced).unwrap();
    let mut config = RoundConfig::default();
    config.surrogate.neighbors = 1;
    let first = search.ask(12, config).unwrap();
    let second = search.ask(13, config).unwrap();
    assert_ne!(first.ticket().region.slot, second.ticket().region.slot);
    assert!(search.tell(&first, f32::NAN, 0.0).is_err());
    assert!(search.buffer(&first).is_ok());
    assert!(search.tell(&second, 3.0, 0.0).unwrap().accepted);
    assert!(search.tell(&first, 2.0, 0.0).unwrap().accepted);
    assert!(search.tell(&first, 2.0, 0.0).is_err());
    assert_eq!(search.best(), (1, 3.0));
    assert_eq!(search.trials(), vec![1, 1]);
    search.restart(0, state(4.0)).unwrap();
    let fresh = search.ask_region(0, 14, config).unwrap();
    assert_eq!(fresh.ticket().region.generation, 1);
    assert!(search.buffer(&first).is_err());
    assert!(search.tell(&fresh, 5.0, 0.0).unwrap().accepted);
}
