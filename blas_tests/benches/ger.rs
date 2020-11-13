use criterion::measurement::Measurement;
use criterion::*;
use ndarray::*;
use ndarray_matops::Ger;
use rand::prelude::*;
use rand_distr::StandardNormal;
use rand_isaac::isaac64::Isaac64Rng;

const SIDES: [usize; 12] = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192];

fn gen_random_numbers<N, Rng: rand::Rng>(num: usize, rng: &mut Rng) -> Vec<N>
where
    StandardNormal: Distribution<N>,
{
    rng.sample_iter(StandardNormal).take(num).collect()
}

fn gen_random_matrix<N, Rng: rand::Rng>(side: usize, is_row_major: bool, rng: &mut Rng) -> Array2<N>
where
    StandardNormal: Distribution<N>,
{
    let data = gen_random_numbers(side * side, rng);
    if is_row_major {
        Array2::from_shape_vec((side, side), data).unwrap()
    } else {
        Array2::from_shape_vec((side, side).f(), data).unwrap()
    }
}

fn gen_random_vector<N, Rng: rand::Rng>(side: usize, rng: &mut Rng) -> Array1<N>
where
    StandardNormal: Distribution<N>,
{
    Array1::from(gen_random_numbers(side, rng))
}

fn bench_ger<N, M: Measurement>(group: &mut BenchmarkGroup<M>, side: &usize, is_row_major: bool)
where
    StandardNormal: Distribution<N>,
    N: LinalgScalar,
{
    let bench_name = if is_row_major {
        "row_major"
    } else {
        "column_major"
    };
    group.bench_with_input(BenchmarkId::new(bench_name, side), side, |b, &side| {
        let mut rng = Isaac64Rng::seed_from_u64(0);
        let mut m: Array2<N> = gen_random_matrix(side, is_row_major, &mut rng);
        let x = gen_random_vector(side, &mut rng);
        let y = gen_random_vector(side, &mut rng);
        let two = N::one() + N::one();
        b.iter(|| m.ger(two, &x, &y));
    });
}

macro_rules! define_bench {
    ($fn:ident, $float:ty) => {
        fn $fn(c: &mut Criterion) {
            let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);
            let mut group = c.benchmark_group(concat!("ger_", stringify!($float)));
            group.plot_config(plot_config);

            #[cfg(feature = "bench_huge")]
            let num_benches = SIDES.len();
            #[cfg(not(feature = "bench_huge"))]
            let num_benches = 8;

            for side in SIDES.iter().take(num_benches) {
                bench_ger::<$float, _>(&mut group, side, true);
                bench_ger::<$float, _>(&mut group, side, false);
            }
        }
    };
}

define_bench! {ger_f32, f32}
define_bench! {ger_f64, f64}

criterion_group!(benches, ger_f32, ger_f64);
criterion_main!(benches);
