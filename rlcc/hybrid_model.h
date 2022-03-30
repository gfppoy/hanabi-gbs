#pragma once

#include "rela/tensor_dict.h"
#include "rlcc/game_sim.h"

namespace approx_search {

inline void addHid(rela::TensorDict& input, const rela::TensorDict& hid) {
  for (auto& kv : hid) {
    auto ret = input.emplace(kv.first, kv.second);
    assert(ret.second);
  }
}

inline void updateHid(rela::TensorDict& output, rela::TensorDict& hid) {
  for (auto& kv : hid) {
    auto it = output.find(kv.first);
    assert(it != output.end());
    auto newHid = it->second;
    assert(newHid.sizes() == kv.second.sizes());
    hid[kv.first] = newHid;
    output.erase(it);
  }
}

class HybridModel {
 public:
  HybridModel(int index)
      : index(index)
      , rlStep_(0) {
  }

  HybridModel(const HybridModel& m)
      : index(m.index)
      , bpModel_(m.bpModel_)
      , bpHid_(m.bpHid_)
      , rlModel_(m.rlModel_)
      , rlHid_(m.rlHid_)
      , rlStep_(m.rlStep_) {
  }

  HybridModel& operator=(const HybridModel& m) {
    assert(index == m.index);
    bpModel_ = m.bpModel_;
    bpHid_ = m.bpHid_;
    rlModel_ = m.rlModel_;
    rlHid_ = m.rlHid_;
    rlStep_ = m.rlStep_;
    return *this;
  }

  void setBpModel(std::shared_ptr<rela::BatchRunner> bpModel) {
    bpModel_ = bpModel;
  }

  void setBpModel(std::shared_ptr<rela::BatchRunner> bpModel, rela::TensorDict bpHid) {
    bpModel_ = bpModel;
    bpHid_ = bpHid;
  }

  void setRlModel(std::shared_ptr<rela::BatchRunner> rlModel, rela::TensorDict rlHid) {
    rlModel_ = rlModel;
    rlHid_ = rlHid;
  }

  rela::Future asyncComputeAction(const GameSimulator& env) const;

  // compute bootstrap target/value using blueprint
  rela::Future asyncComputeTarget(
      const GameSimulator& env, float reward, bool terminal) const;

  // compute priority with rl model
  rela::Future asyncComputePriority(const rela::TensorDict& input) const;

  // observe before act
  void observeBeforeAct(
      const GameSimulator& env, float eps, rela::TensorDict* retFeat = nullptr);

  // observe before act for GBS
  std::vector<torch::Tensor> observeBeforeActGeneralizedBelief(
      const GameSimulator& env, float eps, std::vector<torch::Tensor> aoh, torch::Tensor seq_len, rela::TensorDict* retFeat = nullptr);

  int decideAction(
      const GameSimulator& env, bool verbose, rela::TensorDict* retAction = nullptr);

  int getRlStep() const {
    return rlStep_;
  }

  void setRlStep(int rlStep) {
    assert(rlModel_ != nullptr);
    rlStep_ = rlStep;
  }

  const rela::TensorDict& getBpHid() const {
    return bpHid_;
  }

  void setBpHid(const rela::TensorDict& bpHid) {
    bpHid_ = bpHid;
  }

  const rela::TensorDict& getRlHid() const {
    return rlHid_;
  }

  void setRlHid(const rela::TensorDict& rlHid) {
    rlHid_ = rlHid;
  }

  void setBpModelPy(py::object pyModel) {
    bpModel_->updateModel(pyModel);
  }

  void setRlModelPy(py::object pyModel) {
    rlModel_->updateModel(pyModel);
  }

  const bool hideAction = false;
  const int index;

 private:
  std::shared_ptr<rela::BatchRunner> bpModel_;
  rela::TensorDict bpHid_;
  rela::Future futBp_;

  std::shared_ptr<rela::BatchRunner> rlModel_;
  rela::TensorDict rlHid_;
  rela::Future futRl_;

  int rlStep_;
};
}  // namespace approx_search
