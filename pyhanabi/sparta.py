import os
import signal
import subprocess
import sys
import gc
import random
import argparse
import pprint
import torch
import numpy as np
import torch.nn.functional as F

import utils
import common_utils
from legacy_agent import load_legacy_agent
from utils import load_sad_beliefmodule_model

# c++ backend
import set_path
set_path.append_sys_path()
import rela
import hanalearn
from transformer_embedding import get_model

def run_again(cmd, weight_file, score_sum, num_runs, all_scores, last_pid):
    string_all_scores = ",".join(str(x) for x in all_scores)
    print(string_all_scores)
    subprocess.call(["bash", "-c", "source ~/.profile; " + cmd + " --weight_file " + weight_file + " --score_sum " + str(score_sum) + " --num_runs " + str(num_runs) + " --all_scores " + string_all_scores + " --last_pid " + str(last_pid)])
    exit()

def run(
    seed,
    actors,
    search_actor_idx,
    num_search,
    threshold,
    num_thread,
  #  actor_sus,
):

    belief = get_model(206, 28, 256, 6, 8)
    belief.load_state_dict(torch.load("model$_multi6_234567_bothplayers.pth"))
    belief = belief.to("cuda:1")
    belief.train(False)

    params = {
        "players": str(len(actors)),
        "seed": str(seed),
        "bomb": str(0),
        "hand_size": str(5),
        "random_start_player": str(0),  # do not randomize start_player
    }
    game = hanalearn.GameSimulator(params)
    step = 0
    moves = []

    num_batch = 1
    batchsize = 100

    aoh = torch.zeros((80,1,838)).to("cuda:1")
    seq_len = torch.tensor([1])
    target_seq_size = 6
    nopeak_mask = np.triu(np.ones((1, target_seq_size, target_seq_size)), k=1).astype(np.uint8)
    nopeak_mask = torch.autograd.Variable(torch.from_numpy(nopeak_mask) == 0).to("cuda:1")
    trg = torch.zeros((batchsize, 6)).type(torch.LongTensor).to("cuda:1")


    aoh_for_c = []

    while not game.terminal():
        print("================STEP %d================" % step)
        print(game.state().to_string())

        cur_player = game.state().cur_player()

        obs, cardCount = actors[search_actor_idx].update_belief(game, num_thread)
        src = belief.get_samples_one_player(aoh, seq_len, "cuda:1").type(torch.LongTensor)
        if step < 80:
            aoh[step,0,:] = obs["s"]
            src = belief.get_samples_one_player(aoh, seq_len, "cuda:1").type(torch.LongTensor).repeat(batchsize,1,1).to("cuda:1")
            samples1 = []
            samples2 = []
            samples3 = []
            samples4 = []
            samples5 = []

            for _ in range(num_batch): # 250*num_250s samples per legalMove
                trg[:, :] = 26
                for j in range(5):
                    while True:
                        j_card_dist = np.array(F.softmax(belief(src, trg, None, nopeak_mask)[:,j,:], dim=-1).detach().to("cpu"))
                        temp = np.argmax(np.log(j_card_dist) + np.random.gumbel(size=j_card_dist.shape), axis=1)
                        if 26 not in temp and 27 not in temp:
                            break
                    trg[:,j+1] = torch.tensor(temp)
                    if j == 0:
                        samples1.extend(temp)
                    elif j == 1:
                        samples2.extend(temp)
                    elif j == 2:
                        samples3.extend(temp)
                    elif j == 3:
                        samples4.extend(temp)
                    elif j == 4:
                        samples5.extend(temp)


       # to use BR trained with generalized belief, change below, line 200, as well as line 126 in rlcc/sparta.cc
        using_BR_w_gen_belief = True
        if using_BR_w_gen_belief:
            actors[0].observeGeneralizedBelief(game, aoh_for_c, seq_len.to("cpu"))
        else:
            actors[0].observe(game)
            
        actors[0].observeGeneralizedBelief(game, aoh_for_c, seq_len.to("cpu"))
        
        actors[1].observe(game)

        for i, actor in enumerate(actors):
            print(f"---Actor {i} decide action---")
            action = actor.decide_action(game)
            if i == cur_player:
                move = game.get_move(action)
       
        player_each_rollout = np.random.choice([1,2,3,4,5,6], batchsize*num_batch).tolist()

        #if step < 80 and step > 40:
        if cur_player == search_actor_idx:
            move = actors[search_actor_idx].sparta_search(
                game, move, batchsize*num_batch,#num_search,
                threshold, cardCount, aoh_for_c, seq_len.to("cpu").to(torch.int), player_each_rollout, samples1, samples2, samples3, samples4, samples5)

        print(f"Active Player {cur_player} pick action: {move.to_string()}")
        moves.append(move)
        game.step(move)
        aoh_for_c.append(aoh[step, 0, :].reshape(838))
        step += 1
        seq_len += 1

    print(f"Final Score: {game.get_score()}, Seed: {seed}")
    return moves, game.get_score()


def parse_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("--save_dir", type=str, default="exps/sparta")
    parser.add_argument("--num_runs", type=int, default=0)
    parser.add_argument("--all_scores", type=str, default="")
    parser.add_argument("--score_sum", type=float, default=0)
    parser.add_argument("--last_pid", type=int, default=-420)
    parser.add_argument("--num_search", type=int, default=10000)
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_thread", type=int, default=10)
    parser.add_argument("--weight_file", type=str, default=None)
    parser.add_argument("--num_player", type=int, default=2)
    parser.add_argument("--search_player", type=int, default=0)
    parser.add_argument("--seed", type=int, default=random.randint(1,5000000))
    parser.add_argument("--game_seed", type=int, default=random.randint(1,5000000))
    parser.add_argument("--look_for_finesse", type=int, default=0)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    torch.backends.cudnn.deterministic = True
    args = parse_args()
    if args.last_pid != -420:
        os.kill(args.last_pid, signal.SIGTERM)

    if args.all_scores == "":
        all_scores = []
    else:
        all_scores = [float(s) for s in args.all_scores.split(',')]

    print("Number of runs: " + str(args.num_runs))

    if args.num_runs == 250:
        print("Final average over 250 runs:")
        print(args.score_sum / 250)
        print("Standard error of the mean:")
        sem = 0
        for j in range(len(all_scores)):
            sem += ((args.score_sum/len(all_scores)) - all_scores[j])**2
        sem /= (len(all_scores) - 1)
        sem = sem**0.5 / len(all_scores)
        print(sem)
        exit()

    logger_path = os.path.join(args.save_dir, "train.log")
    sys.stdout = common_utils.Logger(logger_path)

    if "fc_v.weight" in torch.load(args.weight_file).keys():
        #bp, config = utils.load_agent(args.weight_file, {"device": args.device, "vdn": False})
        
        # bp is the agent that will be using search, i.e. the best response agent
        # to use BR trained with generalized belief, change below, line 109, as well as line 126 in rlcc/sparta.cc
        using_BR_w_gen_belief = True
        if using_BR_w_gen_belief:
            bp = load_sad_beliefmodule_model(args.weight_file, args.device)
        else:
            bp, config = load_legacy_agent(args.weight_file)
        
        # bp_partner is the partner we engage in cross-play with
        random_partner = str(np.random.choice([1,8,9,10,11,12,13]))
        bp_partner, _ = load_legacy_agent("models/sad/sad_2p_"+random_partner+".pthw")
        
        # bp_sus actors are partners we will simulate rollouts with in search
        bp_sus2, _ = load_legacy_agent("models/sad/sad_2p_2.pthw")
        bp_sus3, _ = load_legacy_agent("models/sad/sad_2p_3.pthw")
        bp_sus4, _ = load_legacy_agent("models/sad/sad_2p_4.pthw")
        bp_sus5, _ = load_legacy_agent("models/sad/sad_2p_5.pthw")
        bp_sus6, _ = load_legacy_agent("models/sad/sad_2p_6.pthw")
        bp_sus7, _ = load_legacy_agent("models/sad/sad_2p_7.pthw")
        
    else:
        bp = utils.load_supervised_agent(args.weight_file, args.device)
    bp.train(False)
    bp_partner.train(False)

    bp_runners = [rela.BatchRunner(bp, args.device, 1000, ["act"]),
                    rela.BatchRunner(bp_partner, args.device, 1000, ["act"])]

    seed = args.seed
    actors = []
    for i in range(args.num_player):
       # actor = hanalearn.SpartaActor(i, bp_runner, seed)
        actor = hanalearn.SpartaActor(i, bp_runners[i], seed)
        seed += 1
        actors.append(actor)

    # has policy of search policy but id of partner
    bprunner_sus2 = rela.BatchRunner(bp_sus2, args.device, 1000, ["act"])
    actor_sus2 = hanalearn.SpartaActor(1, bprunner_sus2, seed-1)
    bprunner_sus3 = rela.BatchRunner(bp_sus3, args.device, 1000, ["act"])
    actor_sus3 = hanalearn.SpartaActor(1, bprunner_sus3, seed-1)
    bprunner_sus4 = rela.BatchRunner(bp_sus4, args.device, 1000, ["act"])
    actor_sus4 = hanalearn.SpartaActor(1, bprunner_sus4, seed-1)
    bprunner_sus5 = rela.BatchRunner(bp_sus5, args.device, 1000, ["act"])
    actor_sus5 = hanalearn.SpartaActor(1, bprunner_sus5, seed-1)
    bprunner_sus6 = rela.BatchRunner(bp_sus6, args.device, 1000, ["act"])
    actor_sus6 = hanalearn.SpartaActor(1, bprunner_sus6, seed-1)
    bprunner_sus7 = rela.BatchRunner(bp_sus7, args.device, 1000, ["act"])
    actor_sus7 = hanalearn.SpartaActor(1, bprunner_sus7, seed-1)

    actors[args.search_player].set_partners([actors[0], actor_sus2, actor_sus3, actor_sus4, actor_sus5, actor_sus6, actor_sus7], bp)

    for bp_runner in bp_runners:
        bp_runner.start()
    bprunner_sus2.start()
    bprunner_sus3.start()
    bprunner_sus4.start()
    bprunner_sus5.start()
    bprunner_sus6.start()
    bprunner_sus7.start()

    moves, score = run(
         args.game_seed,
         actors,
         args.search_player,
         args.num_search,
         args.threshold,
         args.num_thread,
     )
    print("score: ", score)
    print("Curr avg: " + str((args.score_sum+score)/(args.num_runs+1)))
    if args.num_runs > 1:
        temp_all_scores = all_scores + [score]
        sem = 0
        for j in range(len(temp_all_scores)):
            sem += ((args.score_sum/len(temp_all_scores)) - temp_all_scores[j])**2
        sem /= (len(temp_all_scores) - 1)
        sem = sem**0.5 / len(temp_all_scores)**0.5
        print("SEM: " + str(sem))

    gc.collect()

    run_again("python sparta.py", args.weight_file, args.score_sum+score, args.num_runs+1, all_scores+[score], os.getpid())
