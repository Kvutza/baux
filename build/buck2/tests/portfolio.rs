use baux::semantics::optimizer::{
    RoundId,
    portfolio::{Allocation, Portfolio},
};

fn round(id: u64) -> RoundId {
    RoundId {
        owner: 1,
        proposal: id,
        incumbent: 0,
    }
}

#[test]
fn independent_completion() {
    let mut portfolio = Portfolio::new(vec![((), 0.0), ((), 1.0)], Allocation::Balanced).unwrap();
    let first = portfolio
        .issue(portfolio.choose().unwrap(), round(1))
        .unwrap();
    let second = portfolio
        .issue(portfolio.choose().unwrap(), round(2))
        .unwrap();
    assert!(portfolio.choose().is_err());
    assert!(portfolio.restart(0, (), 2.0).is_err());
    assert!(portfolio.complete(first, f64::NAN, 0.0).is_err());
    portfolio.complete(second, 3.0, 3.0).unwrap();
    assert!(portfolio.complete(second, 4.0, 4.0).is_err());
    portfolio.complete(first, 2.0, 2.0).unwrap();
    assert_eq!(portfolio.trials(), vec![1, 1]);
    assert_eq!(portfolio.best(), (1, 3.0));
    portfolio.restart(0, (), 4.0).unwrap();
    assert_eq!(portfolio.generation(0).unwrap(), 1);
    assert!(portfolio.validate(first).is_err());
    let foreign = Portfolio::new(vec![((), 0.0)], Allocation::Balanced).unwrap();
    assert!(foreign.validate(second).is_err());
}

#[test]
fn retired_completion() {
    let mut portfolio = Portfolio::new(vec![((), 0.0)], Allocation::Balanced).unwrap();
    let ticket = portfolio.issue(0, round(1)).unwrap();
    portfolio.retire(ticket).unwrap();
    assert!(portfolio.validate(ticket).is_err());
    assert_eq!(portfolio.trials(), vec![0]);
    let fresh = portfolio.issue(0, round(2)).unwrap();
    portfolio.complete(fresh, 1.0, 1.0).unwrap();
    assert_eq!(portfolio.trials(), vec![1]);
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
