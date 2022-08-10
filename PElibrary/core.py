import contextlib
import logging
import os
import subprocess
import tempfile

def call_quiet(*args):

    if not len(args):
        raise ValueError("Must supply at least one argument (the command name)")
    try:
        proc = subprocess.Popen(args, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
    except OSError as exc:
        raise RuntimeError("Could not find the executable %r" % args[0]
                           + " -- is it installed correctly?"
                           + "\n(Original error: %s)" % exc)
    out, err = proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError("Subprocess command failed:\n$ %s\n\n%s"
                           % (' '.join(args), err))
    return out


def ensure_path(fname):

    if '/' in os.path.normpath(fname):
        # Ensure the output directory exists
        dname = os.path.dirname(os.path.abspath(fname))
        if dname and not os.path.isdir(dname):
            try:
                os.makedirs(dname)
            except OSError as exc:
                raise OSError("Output path " + fname +
                              " contains a directory " + dname +
                              " that cannot be created: %s" % exc)
    if os.path.isfile(fname):
        # Add an integer suffix to the existing file name
        cnt = 1
        bak_fname = "%s.%d" % (fname, cnt)
        while os.path.isfile(bak_fname):
            cnt += 1
            bak_fname = "%s.%d" % (fname, cnt)
        os.rename(fname, bak_fname)
        logging.info("Moved existing file %s -> %s", fname, bak_fname)
    return True


@contextlib.contextmanager
def temp_write_text(text, mode="w+b"):
    """Save text to a temporary file.

    NB: This won't work on Windows b/c the file stays open.
    """
    with tempfile.NamedTemporaryFile(mode=mode) as tmp:
        tmp.write(text)
        tmp.flush()
        yield tmp.name


def assert_equal(msg, **values):

    ok = True
    key1, val1 = values.popitem()
    msg += ": %s = %r" % (key1, val1)
    for okey, oval in values.items():
        msg += ", %s = %r" % (okey, oval)
        if oval != val1:
            ok = False
    if not ok:
        raise ValueError(msg)


def check_unique(items, title):
    """Ensure all items in an iterable are identical; return that one item."""
    its = set(items)
    assert len(its) == 1, ("Inconsistent %s keys: %s"
                           % (title, ' '.join(map(str, sorted(its)))))
    return its.pop()


def fbase(fname):
    """Strip directory and all extensions from a filename."""
    base = os.path.basename(fname)
    # Gzip extension usually follows another extension
    if base.endswith('.gz'):
        base = base[:-3]
    # Cases to drop more than just the last dot
    known_multipart_exts = (
        '.antitargetcoverage.cnn', '.targetcoverage.cnn',
        '.antitargetcoverage.csv', '.targetcoverage.csv',
        # Pipeline suffixes
        '.recal.bam', '.deduplicated.realign.bam',
    )
    for ext in known_multipart_exts:
        if base.endswith(ext):
            base = base[:-len(ext)]
            break
    else:
        base = base.rsplit('.', 1)[0]
    return base
