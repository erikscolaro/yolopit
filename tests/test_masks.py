"""Block masks on small CNNs (fast, CPU): grouping, leftover block first, export, BN restore."""


def test_block_masks(script):
    script("block_masks.py")


def test_remainder(script):
    script("remainder.py")
