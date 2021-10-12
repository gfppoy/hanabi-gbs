#pragma once

#include "hanabi-learning-environment/hanabi_lib/hanabi_hand.h"
#include "hanabi-learning-environment/hanabi_lib/hanabi_history_item.h"

#include "rlcc/utils.h"
#include "rlcc/hybrid_model.h"

namespace approx_search {

class HandDistribution {
 public:
  HandDistribution() = default;

  void init_(const hle::HanabiGame& game, const std::vector<int>& cardCount);

  void init(const hle::HanabiGame& game, const std::vector<int>& cardCount) {
    init_(game, cardCount);
  }

  int size() const {
    return numOccurances_.size();
  }

  void print(int num) const {
    for (int i = 0; i < std::min(num, (int)allHands_.size()); ++i) {
      for (auto& card : allHands_[i]) {
        std::cout << card.ToString() << ", ";
      }
      std::cout << "occurance: " << numOccurances_[i] << std::endl;
    }
  }

  void updateOccurance(const std::vector<int>& cardCount, bool filterZero);

  void filterMePlayDiscard(
      const hle::HanabiHistoryItem& move,
      const std::vector<int>& cardCount,
      int handSize);

  void filterWithCardKnowledge(const hle::HanabiHand& hand);

  void filterCounterfactual(
      int meIndex,
      int partnerIndex,
      int refAction,
      const HybridModel& partner,
      const hle::HanabiState& state,
      int numThread);

  void filterExact(std::vector<hle::HanabiCardValue> hand, size_t extraHand);

  std::vector<std::vector<hle::HanabiCardValue>> sampleHands(int num, std::mt19937* rng) const;

  const std::vector<hle::HanabiCardValue>& getHand(int i) const {
    return allHands_[i];
  }

  void computeCdf() {
    cdf_ = std::vector<int>(sumOccurance_);
    assert(sumOccurance_ < (int64_t)cdf_.max_size());
    int i = 0;
    for (size_t idx = 0; idx < numOccurances_.size(); ++idx) {
      for (int k = 0; k < numOccurances_[idx]; ++k) {
        cdf_[i] = idx;
        ++i;
      }
    }
    assert(i == sumOccurance_);
  }

 private:
  int numColor_ = 0;
  int numRank_ = 0;

  std::vector<std::vector<hle::HanabiCardValue>> allHands_;
  std::vector<int> numOccurances_;
  int64_t sumOccurance_ = 0;

  std::vector<int> cdf_;
};

void updateBelief(
    const std::unique_ptr<hle::HanabiState>& prevState,
    const hle::HanabiGame& game,
    const std::unique_ptr<hle::HanabiHistoryItem>& lastMove,
    const std::vector<int>& cardCount,  // can be either private/public
    const hle::HanabiHand& myHand,
    HybridModel& partner,
    int myIndex,
    HandDistribution& handDist,
    int numThread,
    bool skipCounterfactual=false);

bool handInBelief(
    const hle::HanabiHand& hand, int playerIdx, const HandDistribution& handDist);

HandDistribution publicToPrivate(
    const HandDistribution& handDist,
    const std::vector<int> cardCount);

}  // namespace approx_search
