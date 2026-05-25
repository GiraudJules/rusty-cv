use std::hint::black_box;
use std::time::Instant;

fn main() {
    let iterations: u64 = std::env::args()
        .nth(1)
        .and_then(|arg| arg.parse().ok())
        .unwrap_or(2_000_000);

    let start = Instant::now();
    let mut checksum: u64 = 0;

    for i in 0..iterations {
        let original_width = black_box(320 + (i % 1600) as u32);
        let original_height = black_box(240 + (i % 1200) as u32);
        let target_width = black_box(640 + (i % 2) as u32 * 640);
        let target_height = black_box(640 + ((i / 2) % 2) as u32 * 640);

        let info = rusty_cv::compute_letterbox(
            original_width,
            original_height,
            target_width,
            target_height,
        )
        .expect("valid dimensions");

        checksum = black_box(checksum).wrapping_add(
            u64::from(info.resized_width)
                + u64::from(info.resized_height)
                + u64::from(info.padding.left)
                + u64::from(info.padding.top),
        );
    }

    let elapsed = start.elapsed();
    black_box(checksum);

    println!("iterations={iterations}");
    println!("elapsed_s={:.6}", elapsed.as_secs_f64());
    println!(
        "ns_per_iter={:.2}",
        elapsed.as_nanos() as f64 / iterations as f64
    );
    println!("checksum={checksum}");
}
