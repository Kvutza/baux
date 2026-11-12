use super::*;
use crate::coordinate::DomainFrame;
use ndarray::array;

#[test]
fn phantom_coords() {
    let bounds = array![[-10.0, 10.0], [0.0, 4.0]];
    let frame = DomainFrame::new(bounds).unwrap();
    let raw = array![[0.0, 2.0], [10.0, 4.0]];
    let dom = Coords::<DomainSpace>::new(raw.view());
    let unit: OwnedCoords<UnitSpace> = frame.encode_unit(dom);
    assert_eq!(unit.as_array(), &array![[0.5, 0.5], [1.0, 1.0]]);
    let back: OwnedCoords<DomainSpace> = frame.decode_domain(unit.borrow());
    assert_eq!(back.as_array(), &raw);

    let embed = (
        PeriodicEmbed {
            low: 0.0,
            high: 6.0,
        },
        (
            SimplexEmbed::<3>,
            (
                LogEmbed {
                    low: 1.0e-4,
                    high: 1.0,
                },
                Categorical::new(5, 42),
            ),
        ),
    );
    assert!(embed.is_tied());
    let samples = vec![
        (1.5, ([0.2, 0.3, 0.5], (1.0e-2, 2usize))),
        (4.5, ([0.6, 0.1, 0.3], (1.0e-3, 4usize))),
    ];
    let unit = encode_batch(&embed, &samples);
    assert_eq!(unit.as_array().ncols(), embed.latent_dim());
    assert!(unit.as_array().iter().all(|&u| (0.0..=1.0).contains(&u)));
    let weights = vec![1.0; embed.latent_dim()];
    let decoded = decode_batch(&embed, unit.borrow(), Some(&weights));
    for ((a0, (s0, (l0, c0))), (a1, (s1, (l1, c1)))) in samples.iter().zip(decoded.iter()) {
        assert!((a0 - a1).abs() < 1.0e-12);
        for (p0, p1) in s0.iter().zip(s1.iter()) {
            assert!((p0 - p1).abs() < 1.0e-12);
        }
        assert!((l0 - l1).abs() < 1.0e-12);
        assert_eq!(c0, c1);
    }
}
