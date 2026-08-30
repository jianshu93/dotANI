use super::{
    GpuPipelineOutput, ani_from_intersection_and_cardinalities, consume_gpu_result_queue, dist,
};
use crate::types::{FileSketch, SketchDist};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Writer that succeeds for a controlled number of writes and then fails
/// deterministically, simulating an output device error (e.g. /dev/full).
struct FailingAfterWriter {
    succeed_writes: usize,
    attempts: Arc<AtomicUsize>,
}

impl std::io::Write for FailingAfterWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        if self.attempts.fetch_add(1, Ordering::SeqCst) >= self.succeed_writes {
            Err(std::io::Error::other("simulated output device failure"))
        } else {
            Ok(buf.len())
        }
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

struct FakePipelineMessage {
    text: String,
    error: Option<anyhow::Error>,
}

impl GpuPipelineOutput for FakePipelineMessage {
    fn output_text(&self) -> Option<&str> {
        if self.error.is_none() {
            Some(self.text.as_str())
        } else {
            None
        }
    }
}

const WRITE_FAILURE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);
const QUEUE_CAPACITY_FOR_TESTS: usize = 2;
const SEND_GRACE_FOR_TESTS: std::time::Duration = std::time::Duration::from_secs(5);

/// Send outcome for fake pipeline stages. Sends are bounded so a shutdown
/// regression can be detected and reported without leaving permanently
/// blocked threads behind in the test process.
enum BoundedSendOutcome {
    Delivered,
    TimedOut,
    Disconnected,
}

fn bounded_send<T>(tx: &crossbeam_channel::Sender<T>, message: T) -> BoundedSendOutcome {
    match tx.send_timeout(message, SEND_GRACE_FOR_TESTS) {
        Ok(()) => BoundedSendOutcome::Delivered,
        Err(crossbeam_channel::SendTimeoutError::Timeout(_)) => BoundedSendOutcome::TimedOut,
        Err(crossbeam_channel::SendTimeoutError::Disconnected(_)) => {
            BoundedSendOutcome::Disconnected
        }
    }
}

#[test]
fn gpu_pipeline_consumer_drains_bounded_queue_after_write_failure() {
    const QUEUE_CAPACITY: usize = 4;
    const PRODUCERS: usize = 3;
    const MESSAGES_PER_PRODUCER: usize = 32;
    const SUCCEED_WRITES: usize = 2;
    const TOTAL_MESSAGES: usize = PRODUCERS * MESSAGES_PER_PRODUCER;

    let (result_tx, result_rx) = crossbeam_channel::bounded::<FakePipelineMessage>(QUEUE_CAPACITY);
    let cancel = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let write_attempts = Arc::new(AtomicUsize::new(0));
    let blocked_sends = Arc::new(AtomicUsize::new(0));
    let (done_tx, done_rx) =
        std::sync::mpsc::channel::<(usize, usize, usize, bool, Option<anyhow::Error>)>();

    let cancel_for_child = Arc::clone(&cancel);
    let attempts_for_child = Arc::clone(&write_attempts);
    let blocked_for_child = Arc::clone(&blocked_sends);
    let child = std::thread::spawn(move || {
        let cancel = cancel_for_child;
        let mut received = 0usize;
        let mut first_error = None;

        std::thread::scope(|scope| {
            for producer in 0..PRODUCERS {
                let result_tx = result_tx.clone();
                let blocked = Arc::clone(&blocked_for_child);
                scope.spawn(move || {
                    for message in 0..MESSAGES_PER_PRODUCER {
                        let message = FakePipelineMessage {
                            text: format!("producer {producer} message {message}\n"),
                            error: None,
                        };
                        match bounded_send(&result_tx, message) {
                            BoundedSendOutcome::Delivered => {}
                            BoundedSendOutcome::TimedOut => {
                                blocked.fetch_add(1, Ordering::SeqCst);
                                break;
                            }
                            BoundedSendOutcome::Disconnected => break,
                        }
                    }
                });
            }
            drop(result_tx);

            let mut writer = FailingAfterWriter {
                succeed_writes: SUCCEED_WRITES,
                attempts: Arc::clone(&attempts_for_child),
            };
            first_error = consume_gpu_result_queue(
                &result_rx,
                Some(&mut writer),
                &cancel,
                |message, _write_elapsed| match message {
                    FakePipelineMessage { error: Some(e), .. } => Some(e),
                    FakePipelineMessage { .. } => {
                        received += 1;
                        None
                    }
                },
            );
        });

        let outcome = (
            received,
            write_attempts.load(Ordering::SeqCst),
            blocked_sends.load(Ordering::SeqCst),
            cancel.load(Ordering::SeqCst),
            first_error,
        );
        done_tx
            .send(outcome)
            .expect("supervisor dropped the outcome receiver before shutdown");
    });

    let (received, attempts, blocked, cancelled, first_error) = done_rx
        .recv_timeout(WRITE_FAILURE_TIMEOUT)
        .expect("consumer deadlocked on bounded result queue instead of draining it");
    child
        .join()
        .expect("supervised pipeline thread panicked during drain");

    assert_eq!(
        received, TOTAL_MESSAGES,
        "consumer must drain every message so all blocked senders can unblock and join"
    );
    assert_eq!(
        attempts,
        SUCCEED_WRITES + 1,
        "consumer must stop attempting writes after the first failure"
    );
    assert_eq!(
        blocked, 0,
        "no sender may stay blocked until the send grace expires; the consumer must unblock them"
    );
    assert!(
        cancelled,
        "write failure must set the shared cancellation flag immediately"
    );
    let error = first_error
        .expect("consumer must return the original write error after shutdown completes");
    assert!(
        error.to_string().contains("failed to write ANI batch"),
        "unexpected write error context: {error:#}"
    );
    let root_cause = error
        .root_cause()
        .downcast_ref::<std::io::Error>()
        .expect("original io error must be preserved as the root cause");
    assert_eq!(root_cause.kind(), std::io::ErrorKind::Other);
    assert_eq!(root_cause.to_string(), "simulated output device failure");
}

#[test]
fn gpu_pipeline_consumer_keeps_first_pipeline_error_over_write_failure() {
    let (result_tx, result_rx) =
        crossbeam_channel::bounded::<FakePipelineMessage>(QUEUE_CAPACITY_FOR_TESTS);
    let cancel = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let write_attempts = Arc::new(AtomicUsize::new(0));
    let mut received = 0usize;

    result_tx
        .send(FakePipelineMessage {
            text: String::new(),
            error: Some(anyhow::anyhow!("CUDA worker failure")),
        })
        .expect("bounded queue has room for the pipeline error");
    result_tx
        .send(FakePipelineMessage {
            text: "batch whose write will fail\n".to_string(),
            error: None,
        })
        .expect("bounded queue has room for the failing batch");
    drop(result_tx);

    let mut writer = FailingAfterWriter {
        succeed_writes: 0,
        attempts: Arc::clone(&write_attempts),
    };
    let first_error = consume_gpu_result_queue(
        &result_rx,
        Some(&mut writer),
        &cancel,
        |message, _write_elapsed| match message {
            FakePipelineMessage { error: Some(e), .. } => Some(e),
            FakePipelineMessage { .. } => {
                received += 1;
                None
            }
        },
    );

    assert_eq!(received, 1, "all queued messages must still be drained");
    assert_eq!(
        write_attempts.load(Ordering::SeqCst),
        1,
        "the batch after the pipeline error is written once; further writes stop after failure"
    );
    assert!(
        cancel.load(Ordering::SeqCst),
        "pipeline errors must keep setting the shared cancellation flag"
    );
    let error = first_error.expect("an error must be reported");
    assert!(
        error.to_string().contains("CUDA worker failure"),
        "the earlier pipeline error must not be overwritten by the write error, got: {error:#}"
    );
}

#[test]
#[test]
fn ani_clamps_estimated_jaccard_overshoot_to_100() {
    let ani = ani_from_intersection_and_cardinalities(120.0, 100.0, 100.0, 16);
    assert!(
        (ani - 100.0).abs() < f32::EPSILON,
        "expected overshot Jaccard estimate to produce 100 ANI, got {ani}"
    );
}

#[test]
fn dist_validates_public_boundary_parameters_before_loading_files() {
    let cases = [
        (
            SketchDist {
                ani_threshold: f32::NAN,
                ..SketchDist::default()
            },
            "ANI threshold",
        ),
        (
            SketchDist {
                threads: 0,
                ..SketchDist::default()
            },
            "thread count",
        ),
        (SketchDist::default(), "output path"),
    ];

    for (mut params, expected) in cases {
        let error = dist(&mut params).unwrap_err().to_string();
        assert!(error.contains(expected), "unexpected error: {error}");
    }
}

fn test_filesketch(file_str: &str, hv: Vec<i32>) -> FileSketch {
    FileSketch {
        ksize: 1,
        scaled: 1,
        canonical: true,
        seed: 123,
        hv_d: hv.len(),
        hv_quant_bits: 0,
        hv_norm_2: 0,
        file_str: file_str.to_string(),
        hv,
    }
}

// --- lazy resident host-matrix flattening regression tests ---

#[cfg(feature = "cuda")]
#[test]
fn resident_matrix_bytes_checked_and_overflow() {
    let (upload_bytes, required_bytes) =
        super::resident_matrix_bytes(3, 4, 2, 512, 512).expect("small plan must not overflow");
    assert_eq!(
        upload_bytes,
        3 * 4 * std::mem::size_of::<i32>() + 2 * std::mem::size_of::<f64>()
    );
    assert_eq!(
        required_bytes,
        upload_bytes + 512 * 512 * std::mem::size_of::<i64>() + super::RESIDENT_SAFETY_BYTES
    );
    // Any overflowed component must report the resident path as infeasible.
    assert!(super::resident_matrix_bytes(usize::MAX, usize::MAX, 1, 512, 512).is_none());
}

#[cfg(feature = "cuda")]
#[test]
fn resident_flat_if_feasible_gates_flatten_and_shares_one_matrix() {
    let sketches = vec![
        test_filesketch("s0", vec![1, 2, 3, 4]),
        test_filesketch("s1", vec![5, 6, 7, 8]),
    ];
    let (_, required_bytes) = super::resident_matrix_bytes(sketches.len(), 4, 2, 512, 512)
        .expect("small plan must not overflow");
    let cell = std::sync::OnceLock::<(Vec<i32>, u128)>::new();

    // Equal and insufficient capacity leave the shared cell empty.
    assert!(
        super::resident_flat_if_feasible(required_bytes, required_bytes, &cell, &sketches)
            .is_none()
    );
    assert!(super::resident_flat_if_feasible(0, required_bytes, &cell, &sketches).is_none());
    assert!(cell.get().is_none());

    // Feasible scoped workers all share the one initialized matrix.
    const WORKERS: usize = 4;
    let mut observed: Vec<&[i32]> = Vec::new();
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..WORKERS)
            .map(|_| {
                let cell = &cell;
                let sketches = &sketches;
                scope.spawn(move || {
                    super::resident_flat_if_feasible(
                        required_bytes + 1,
                        required_bytes,
                        cell,
                        sketches,
                    )
                })
            })
            .collect();
        for handle in handles {
            observed.push(
                handle
                    .join()
                    .expect("resident worker panicked")
                    .expect("feasible worker must see the resident matrix"),
            );
        }
    });

    assert_eq!(observed.len(), WORKERS);
    let expected: &[i32] = &[1, 2, 3, 4, 5, 6, 7, 8];
    assert_eq!(observed[0], expected);
    assert!(
        observed.iter().all(|flat| std::ptr::eq(*flat, observed[0])),
        "every feasible worker must share the one flattened matrix"
    );
}
