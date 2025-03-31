import pandas
import lockfile

import bz2
import os
import os.path

from pgn_parsering import GamesFile
from multiproc import Multiproc, MultiprocWorker, MultiprocIterable

def main():
    df_targets = pandas.read_csv("./player_count.csv.bz2")
    players_top_df = df_targets.sort_values("count", ascending=False).head(3000)
    targets = set(players_top_df['player'])

    # os.makedirs("./player_pgns", exist_ok=True)

    multiProc = Multiproc(48)
    multiProc.reader_init(Files_lister, ["./lichess_db_standard_rated_2021-02.pgn"])
    multiProc.processor_init(Games_processor, targets, "./player_pgns", False)
    multiProc.run()

class Files_lister(MultiprocIterable):
    def __init__(self, inputs):
        self.inputs = list(inputs)
    def __next__(self):
        try:
            return self.inputs.pop()
        except IndexError:
            raise StopIteration

class Games_processor(MultiprocWorker):
    def __init__(self, targets, output_dir, exclude_bullet):
        self.output_dir = output_dir
        self.targets = targets
        self.exclude_bullet = exclude_bullet

        self.c = 0

    def __call__(self, path):
        games = GamesFile(path)
        self.c = 0
        for i, (d, s) in enumerate(games):
            if self.exclude_bullet and 'Bullet' in d['Event']:
                continue
            else:
                if d['White'] in self.targets:
                    self.write_player(d['White'], s, True)
                    self.c += 1
                if d['Black'] in self.targets:
                    self.write_player(d['Black'], s, False)
                    self.c += 1
            if i % 10000 == 0:
                print(i)

    def write_player(self, p_name, s, is_white):
        if is_white:
            path = f"./players/{p_name}"
            p_path = os.path.join(path, f"white.pgn.bz2")
        else:
            path = f"./players/{p_name}"
            p_path = os.path.join(path, f"black.pgn.bz2")
        os.makedirs(path, exist_ok=True)
        
        lock_path = p_path + '.lock'
        lock = lockfile.FileLock(lock_path)
        with lock:
            with bz2.open(p_path, 'at') as f:
                f.write(s)


if __name__ == '__main__':
    main()
