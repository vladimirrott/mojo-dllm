"""Read-only memory maps of model files.

A 5 GB GGUF file is mapped, never read into the heap. The kernel faults pages
in as the first forward pass touches them, and quantized weights stay in the
page cache in their on-disk layout.

`mmap` reports failure as (void *)-1, so the result is compared as an integer
before anything turns it into a pointer.
"""

from std.ffi import c_int, c_size_t, external_call

from mojo_dllm.sys.mem import U8Ptr

comptime O_RDONLY = 0
comptime PROT_READ = 1
comptime MAP_PRIVATE = 2
comptime SEEK_END = 2
comptime MADV_WILLNEED = 3
comptime MADV_DONTNEED = 4
comptime MADV_POPULATE_READ = 22
comptime PAGE = 4096


struct Mapping(Movable):
    """A whole file mapped read-only. Unmapped when dropped."""

    var addr: Int
    var size: Int
    var path: String

    def __init__(out self, path: String) raises:
        self.path = path
        var cpath = path
        var fd = external_call["open", c_int, num_fixed_args=2](
            cpath.as_c_string_span(), c_int(O_RDONLY), c_int(0)
        )
        _ = cpath^
        if Int(fd) < 0:
            raise Error("cannot open file: " + path)
        var end = Int(external_call["lseek", Int](fd, Int(0), c_int(SEEK_END)))
        if end <= 0:
            _ = external_call["close", c_int](fd)
            raise Error("file is empty or unreadable: " + path)
        var addr = Int(
            external_call["mmap", Int](
                Int(0),
                c_size_t(end),
                c_int(PROT_READ),
                c_int(MAP_PRIVATE),
                fd,
                Int(0),
            )
        )
        _ = external_call["close", c_int](fd)
        if addr == -1 or addr == 0:
            raise Error("mmap failed for " + path)
        self.addr = addr
        self.size = end

    def __deinit__(deinit self):
        _ = external_call["munmap", c_int](self.addr, c_size_t(self.size))

    def ptr(self) -> U8Ptr:
        return U8Ptr(unsafe_from_address=self.addr)

    def will_need(self):
        """Ask the kernel to start reading the whole file ahead of use."""
        _ = external_call["madvise", c_int](
            self.addr, c_size_t(self.size), c_int(MADV_WILLNEED)
        )

    def drop(self, addr: Int, size: Int):
        """Release the resident pages of [addr, addr+size), shrunk to whole pages.

        Used after a tensor has been repacked into the heap: the mapped bytes
        will not be read again, and leaving them resident would double-count
        the model in RSS. A later read faults the page back in from the file.
        """
        var lo = (addr + PAGE - 1) // PAGE * PAGE
        var hi = (addr + size) // PAGE * PAGE
        if hi > lo:
            _ = external_call["madvise", c_int](
                lo, c_size_t(hi - lo), c_int(MADV_DONTNEED)
            )

    def populate(self, addr: Int, size: Int):
        """Fault in [addr, addr+size) with one call (Linux 5.14+).

        Repacking reads every byte of a tensor once. Without this the kernel
        takes one page fault per 4 KiB page; MADV_POPULATE_READ reads the
        range in bulk. Failure (an older kernel) only costs speed.
        """
        var lo = addr // PAGE * PAGE
        var hi = (addr + size + PAGE - 1) // PAGE * PAGE
        if hi > lo:
            _ = external_call["madvise", c_int](
                lo, c_size_t(hi - lo), c_int(MADV_POPULATE_READ)
            )
