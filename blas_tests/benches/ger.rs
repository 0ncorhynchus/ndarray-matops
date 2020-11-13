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

fn ger(c: &mut Criterion) {
    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);
    let mut group = c.benchmark_group("matmul");
    group.plot_config(plot_config);

    #[cfg(feature = "bench_huge")]
    let num_benches = SIDES.len();
    #[cfg(not(feature = "bench_huge"))]
    let num_benches = 8;

    for side in SIDES.iter().take(num_benches) {
        macro_rules! bench {
            ($float:ty, $is_row_major:expr) => {{
                let bench_name = if $is_row_major {
                    concat!(stringify!($float), "_row_major")
                } else {
                    concat!(stringify!($float), "_column_major")
                };
                group.bench_with_input(BenchmarkId::new(bench_name, side), side, |b, &side| {
                    let mut rng = Isaac64Rng::seed_from_u64(0);
                    let mut m: Array2<$float> = gen_random_matrix(side, $is_row_major, &mut rng);
                    let x = gen_random_vector(side, &mut rng);
                    let y = gen_random_vector(side, &mut rng);
                    b.iter(|| m.ger(2.0, &x, &y));
                });
            }};
        }

        bench! {f32, true}
        bench! {f32, false}
        bench! {f64, true}
        bench! {f64, false}
    }
}

criterion_group!(benches, ger);
criterion_main!(benches);
