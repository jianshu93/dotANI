use anyhow::{Context, Result, anyhow, bail};
use crossbeam_channel::{Receiver, Sender, bounded};
use needletail::{Sequence, parse_fastx_file};
use std::path::Path;

#[derive(Debug)]
pub struct MergedSequence {
    pub sequence: Vec<u8>,
    pub input_bases: usize,
}

pub struct ReaderGate {
    tokens: Sender<()>,
    releases: Receiver<()>,
}

pub struct ReaderPermit {
    releases: Sender<()>,
}

impl ReaderGate {
    pub fn new(limit: usize) -> Self {
        assert!(limit > 0, "reader gate limit must be greater than zero");

        let (tokens, releases) = bounded(limit);
        for _ in 0..limit {
            tokens.send(()).expect("reader gate token channel closed");
        }

        Self { tokens, releases }
    }

    pub fn acquire(&self) -> ReaderPermit {
        self.releases
            .recv()
            .expect("reader gate token channel closed");
        ReaderPermit {
            releases: self.tokens.clone(),
        }
    }
}

impl Drop for ReaderPermit {
    fn drop(&mut self) {
        let _ = self.releases.send(());
    }
}

// Read normalized records into one sequence, with N separators only between records.
pub fn read_merge_seq(file_name: &Path) -> Result<MergedSequence> {
    let mut fna_seqs = Vec::<u8>::new();
    let mut input_bases = 0usize;
    let mut record_count = 0usize;

    let mut fastx_reader = parse_fastx_file(file_name)
        .map_err(|e| anyhow!("failed to open FASTA/FASTQ {}: {e}", file_name.display()))?;
    while let Some(record) = fastx_reader.next() {
        let seqrec = record.with_context(|| {
            format!(
                "failed to parse FASTA/FASTQ record in {}",
                file_name.display()
            )
        })?;
        let norm_seq = seqrec.normalize(false);

        if record_count > 0 {
            fna_seqs.push(b'N');
        }
        fna_seqs.extend_from_slice(norm_seq.as_ref());
        input_bases = input_bases
            .checked_add(norm_seq.len())
            .ok_or_else(|| anyhow!("normalized input length overflows usize"))?;
        record_count += 1;
    }

    if record_count == 0 {
        bail!(
            "FASTA/FASTQ file {} contains no records",
            file_name.display()
        );
    }

    Ok(MergedSequence {
        sequence: fna_seqs,
        input_bases,
    })
}

#[cfg(test)]
mod tests {
    use super::{ReaderGate, read_merge_seq};
    use std::fs;
    use std::path::PathBuf;
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };
    use std::time::Duration;

    fn test_file(name: &str, contents: &[u8]) -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "dotani_fastx_{}_{}_{}.fna",
            std::process::id(),
            name,
            std::thread::current().name().unwrap_or("unnamed")
        ));
        fs::write(&path, contents).unwrap();
        path
    }

    #[test]
    fn merged_sequence_separates_records_without_counting_synthetic_n() {
        let path = test_file("merged", b">one\nacgt\n>two\ntt\n");

        let merged = read_merge_seq(&path).unwrap();

        assert_eq!(merged.sequence, b"ACGTNTT");
        assert_eq!(merged.input_bases, 6);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn malformed_fastx_returns_an_error() {
        let path = test_file("malformed", b"@one\nACGT\n+\n!!\n");

        let error = read_merge_seq(&path).expect_err("malformed FASTQ should fail");

        assert!(error.to_string().contains("parse"));
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn reader_gate_limits_concurrent_readers() {
        let limit = 2;
        let gate = Arc::new(ReaderGate::new(limit));
        let active = Arc::new(AtomicUsize::new(0));
        let max_active = Arc::new(AtomicUsize::new(0));

        std::thread::scope(|scope| {
            for _ in 0..8 {
                let gate = Arc::clone(&gate);
                let active = Arc::clone(&active);
                let max_active = Arc::clone(&max_active);

                scope.spawn(move || {
                    let _permit = gate.acquire();
                    let now_active = active.fetch_add(1, Ordering::SeqCst) + 1;
                    max_active.fetch_max(now_active, Ordering::SeqCst);
                    std::thread::sleep(Duration::from_millis(10));
                    active.fetch_sub(1, Ordering::SeqCst);
                });
            }
        });

        assert!(max_active.load(Ordering::SeqCst) <= limit);
    }

    #[test]
    fn reader_gate_permit_drop_releases_next_acquire() {
        let gate = Arc::new(ReaderGate::new(1));
        let first = gate.acquire();

        let acquired = Arc::new(AtomicUsize::new(0));
        std::thread::scope(|scope| {
            let gate = Arc::clone(&gate);
            let acquired_in_thread = Arc::clone(&acquired);
            scope.spawn(move || {
                let _permit = gate.acquire();
                acquired_in_thread.store(1, Ordering::SeqCst);
            });

            std::thread::sleep(Duration::from_millis(10));
            assert_eq!(acquired.load(Ordering::SeqCst), 0);
            drop(first);
        });

        assert_eq!(acquired.load(Ordering::SeqCst), 1);
    }
}
