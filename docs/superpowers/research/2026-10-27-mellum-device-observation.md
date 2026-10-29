# macOS device-read observation research brief

**Date:** 2026-10-27
**Question:** How can the next CPU-only diagnostic distinguish hot reads from
storage activity without assuming cache-control flags prove SSD service?
**Scope:** Read-only inspection of installed macOS documentation, device topology
and Apple kernel reference source. No new payload reads, inference or benchmark.

## Findings

1. **Claim:** Successful cache controls do not establish a physical read.
   - **Evidence:** Installed `fcntl(2)` describes F_NOCACHE as turning caching
     off/on and F_RDAHEAD as controlling read-ahead. The installed SDK labels
     F_NOCACHE as descriptor-scoped, F_GLOBAL_NOCACHE as global for the file,
     and F_NOCACHE_EXT's relaxed alignment as applying to writes.
   - **Source:** Apple, installed `man 2 fcntl`; SDK `usr/include/sys/fcntl.h`,
     lines 257, 264, 270, 340, accessed 2026-10-27. SDK root:
     `/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk`.
   - **Assessment:** Fact about the documented interface; no cold-read guarantee.

2. **Claim:** Even the reference direct-read path consults cached pages.
   - **Evidence:** `cluster_read_direct` sets CL_NOCACHE at lines 5426–5427,
     handles misalignment through the cache at 5504–5528, and explicitly
     “First look[s] for pages already in the cache” at 5576–5587, calling
     `cluster_copy_ubc_data_internal`. `cluster_io_type` chooses IO_DIRECT
     at 6264–6265 for eligible ordinary buffers; this does not bypass that check.
   - **Source:** [vfs_cluster.c](https://github.com/apple-oss-distributions/xnu/blob/f6217f891ac0bb64f3d375211650a4c1ff8ca1ea/bsd/vfs/vfs_cluster.c#L5576)
     — Apple OSS, accessed 2026-10-27, pinned reference revision
     `f6217f891ac0bb64f3d375211650a4c1ff8ca1ea`, file SHA-256
     `3d210d3c725d63f11f85b774ec9ea77b80125a39333ed96c561a99157b930289`.
   - **Assessment:** Fact about reference source, not verified running-kernel
     provenance. Alignment changes alone cannot be assumed to force SSD reads
     (inference). This does not prove the cause of the S3a timings.

3. **Claim:** Read-specific cumulative block-driver counters are accessible.
   - **Evidence:** `ioreg -a -r -c IOBlockStorageDriver` returned Statistics
     including Bytes (Read), Bytes (Write), Operations (Read), errors and retries
     without privilege. The installed header defines Bytes (Read) as bytes
     since driver instantiation, and read operations separately. One lifetime
     observation was 236,601,069,568 read bytes; it is not an experiment delta.
   - **Source:** Apple installed SDK,
     `System/Library/Frameworks/IOKit.framework/Versions/A/Headers/storage/IOBlockStorageDriver.h`,
     lines 44–68, 141–152, 197–208; local read-only `ioreg`, accessed 2026-10-27.
   - **Assessment:** Fact. These counters describe device-wide driver activity,
     not a process/file attribution or NAND-internal bytes.

4. **Claim:** This host's measured files can be mapped to a specific driver.
   - **Evidence:** Read-only diskutil metadata maps APFS data volume disk3s5
     through container disk3, UUID `8C90DA7F-D18C-44E8-86D1-0F854C355C46`,
     to its sole physical store disk0s2 and whole disk0. `ioreg -a -l -r -c
     IOBlockStorageDriver` exposes the direct child IOMedia with BSD Name disk0,
     Whole true, Size 2,001,111,162,880 and Preferred Block Size 4096 under
     driver entry 4294971011; media entry is 4294971012. The other driver has
     no matching child. Descendant properties require `-l`; root statistics
     alone do not establish this mapping.
   - **Source:** Apple `diskutil apfs list -plist`, `ioreg` commands above,
     accessed 2026-10-27. Prior device metadata retained with
     [S3a evidence](2026-10-27-mellum-storage-evidence.md).
   - **Assessment:** Fact about this boot/inspection. Future execution must
     rediscover topology; registry IDs are not permanent device identities.

5. **Claim:** Existing role reads are not naturally page/block aligned.
   - **Evidence:** Existing verified metadata gives role length 2,193,408,
     expert length 6,580,224 and GGUF data start 3,589,408 bytes. Modulo 4096,
     these are 2048, 2048 and 1312; modulo this host's 16,384-byte page size,
     they are 14336, 10240 and 1312. Current workers use malloc, without an
     explicit page alignment contract.
   - **Source:** Checked S3a index/plan metadata and `tools/mellum_read_run.c`;
     local `sysconf(SC_PAGE_SIZE)`, accessed 2026-10-27.
   - **Assessment:** Fact. Not evidence that an aligned variant would force
     storage; do not build that variant before observing the current path.

6. **Claim:** iostat alone is insufficient for the intended short read test.
   - **Evidence:** Its first display averages since boot; later displays cover
     intervals. Standard transfer totals mix reads and writes. `-I` changes
     totals versus rates, not file attribution.
   - **Source:** Apple installed `man 8 iostat`, accessed 2026-10-27,
     DESCRIPTION and OPTIONS.
   - **Assessment:** Fact about interface; recommendation to retain read-specific
     counter snapshots and independently timed brackets instead.

## Open questions and limits

- No experiment deltas have been measured. S3a's 29–78 GB/s remains warm-read
  evidence with unverified physical traffic.
- Device-wide counters cannot rule out a coincident background workload.
  Quiet adjacent controls can flag contamination but cannot establish causality.
- APFS compression, read amplification and storage-controller caches complicate
  attribution. Block-driver traffic is not a measurement of NAND service.
- No SSD miss stall, streamed inference speed or physical 8 GB feasibility is known.

## Recommendation

High confidence: add observation before changing the read path. Moderate
confidence: a small, separately reserved diagnostic with native timing gates,
raw device counters and adjacent idle controls can identify negligible activity
or activity consistent with the requested byte volume. Preserve inconclusive
results and stop after the declared budget. No aligned variant, eviction scheme,
cache purge or larger file is justified by this research alone. See the
[proposed design](../specs/2026-10-27-mellum-device-observation-design.md).
