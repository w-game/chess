import os
import subprocess

import shutil
import glob

def move_to_player(base_path, path):
    target_path = f"{base_path}/{path}"
    os.makedirs(target_path, exist_ok=True)
    supervised_src = os.path.join(base_path, "supervised-0")
    supervised_dest = os.path.join(target_path, "supervised-0")
    
    shutil.move(supervised_src, supervised_dest)

    pgn_files = glob.glob(f"{base_path}/*.pgn")
    for file_path in pgn_files:
        os.remove(file_path)

def worker(player_name):
    player_dir = f"./players/{player_name}"

    for color in ["white", "black"]:
        target_dir = f"{player_dir}/{color}"
        if os.path.exists(target_dir) and os.path.isdir(target_dir):
            continue
            # shutil.rmtree(target_dir)
            
        subprocess.run(f"bzcat {color}.pgn.bz2 | pgn-extract -7 -C -N -#100",
                       shell=True, text=True, check=True, cwd=player_dir)

        subprocess.run(f"trainingdata-tool -v 1.pgn",
                       shell=True, text=True, check=True, cwd=player_dir)

        move_to_player(player_dir, color)


if __name__ == "__main__":
    # num_workers = int(os.cpu_count() / 2)
    # print(f"Using {num_workers} workers")

    players = os.listdir("./players")
    
    for player in players:
        worker(player)

    # with multiprocessing.Pool(processes=num_workers) as pool:
    #     results = pool.map(worker, players)
    
