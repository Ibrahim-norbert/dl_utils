import os
import shutil

class FileSystem:
    def __init__(self, root_dir):
        self.root_dir = root_dir


    def removeSuperflousDirs(self, dry_run : bool,  keep_exts=(".json", ".csv", ".png", ".ckpt", ".pth")):

        def has_kept_file(path):
            """True if any file under path (recursively) has one of keep_exts."""
            for _, _, files in os.walk(path):
                if any(f.lower().endswith(keep_exts) for f in files):
                    return True
            return False


        # topdown=True visits parents before children (shallowest first)
        for dirpath, dirnames, _ in os.walk(self.root_dir, topdown=True):
            for d in sorted(dirnames):
                path = os.path.join(dirpath, d)
                if os.path.islink(path):
                    continue
                if not has_kept_file(path):
                    print(("Would delete: " if dry_run else "Deleting: ") + path)
                    if not dry_run:
                        shutil.rmtree(path)
                    dirnames.remove(d)  # prune: don't descend into it
