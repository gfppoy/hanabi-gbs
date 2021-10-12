#pragma once

#include "rlcc/game_sim.h"
#include "rlcc/hybrid_model.h"
#include "rlcc/hand_dist.h"

namespace sparta {

class SpartaActor {
 public:
  SpartaActor(
      int index,
      std::shared_ptr<rela::BatchRunner> bpRunner,
      int seed)
      : index(index)
      , rng_(seed)
      , prevModel_(index)
      , model_(index) {
    assert(bpRunner != nullptr);
    model_.setBpModel(bpRunner, getH0(*bpRunner, 1));
  }

  void setPartners(std::vector<std::shared_ptr<SpartaActor>> partners) {
    partners_ = std::move(partners);
  }

  void updateBelief(const approx_search::GameSimulator& env, int numThread) {
    assert(callOrder_ == 0);
    ++callOrder_;

    const auto& state = env.state();
    int curPlayer = state.CurPlayer();
    int numPlayer = env.game().NumPlayers();
    assert((int)partners_.size() == numPlayer);
    int prevPlayer = (curPlayer - 1 + numPlayer) % numPlayer;
    std::cout << "prev player: " << prevPlayer << std::endl;

    auto [obs, lastMove, cardCount, myHand] =
        observeForSearch(env.state(), index, hideAction, false);

    approx_search::updateBelief(
        prevState_,
        env.game(),
        lastMove,
        cardCount,
        myHand,
        partners_[prevPlayer]->prevModel_,
        index,
        handDist_,
        numThread);
  }

  void observe(const approx_search::GameSimulator& env) {
    // assert(callOrder_ == 1);
    ++callOrder_;

    const auto& state = env.state();
    model_.observeBeforeAct(env, 0);

    if (prevState_ == nullptr) {
      prevState_ = std::make_unique<hle::HanabiState>(state);
    } else {
      *prevState_ = state;
    }
  }

  int decideAction(const approx_search::GameSimulator& env) {
    // assert(callOrder_ == 2);
    callOrder_ = 0;

    prevModel_ = model_;  // this line can only be in decide action
    return model_.decideAction(env, false);
  }

  // should be called after decideAction?
  hle::HanabiMove spartaSearch(
      const approx_search::GameSimulator& env,
      hle::HanabiMove bpMove,
      int numSearch,
      float threshold);

  const int index;
  const bool hideAction = false;

 private:
  mutable std::mt19937 rng_;

  approx_search::HybridModel prevModel_;
  approx_search::HybridModel model_;
  approx_search::HandDistribution handDist_;

  std::vector<std::shared_ptr<SpartaActor>> partners_;
  std::unique_ptr<hle::HanabiState> prevState_ = nullptr;

  int callOrder_ = 0;
};

}  // namespace approx_search
