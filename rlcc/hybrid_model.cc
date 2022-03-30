#include <torch/extension.h>
#include "rlcc/hybrid_model.h"
#include <string>

namespace approx_search {

rela::Future HybridModel::asyncComputeAction(const GameSimulator& env) const {
  auto input = observe(env.state(), index, false, std::vector<int>(), std::vector<int>(), hideAction, true, true);
  // auto input = observe(env.state(), index, hideAction);

  if (rlStep_ > 0) {
    addHid(input, rlHid_);
    input["eps"] = torch::tensor(std::vector<float>{0});
    return rlModel_->call("act", input);
  } else {
    addHid(input, bpHid_);
    return bpModel_->call("act", input);
  }
}

// compute bootstrap target/value using blueprint
rela::Future HybridModel::asyncComputeTarget(
    const GameSimulator& env, float reward, bool terminal) const {
  auto feat = observe(env.state(), index, false, std::vector<int>(), std::vector<int>(), false, true, true);
  // auto feat = observe(env.state(), index, false);

  feat["reward"] = torch::tensor(reward);
  feat["terminal"] = torch::tensor((float)terminal);
  addHid(feat, bpHid_);
  return bpModel_->call("compute_target", feat);
}

// compute priority with rl model
rela::Future HybridModel::asyncComputePriority(const rela::TensorDict& input) const {
  assert(rlModel_ != nullptr);
  return rlModel_->call("compute_priority", input);
}

// observe before act
void HybridModel::observeBeforeAct(
    const GameSimulator& env, float eps, rela::TensorDict* retFeat) {
  auto feat = observe(env.state(), index, false, std::vector<int>(), std::vector<int>(), hideAction, true, true);
  // auto feat = observe(env.state(), index, hideAction);

  if (retFeat != nullptr) {
    *retFeat = feat;
  }

  // forward bp regardless of whether rl is used
  {
    auto input = feat;
    addHid(input, bpHid_);
    futBp_ = bpModel_->call("act", input);
  }

  // maybe forward rl
  if (rlStep_ > 0) {
    feat["eps"] = torch::tensor(std::vector<float>{eps});
    auto input = feat;
    addHid(input, rlHid_);
    futRl_ = rlModel_->call("act", input);
  }
}

// observe before act for GBS
std::vector<torch::Tensor> HybridModel::observeBeforeActGeneralizedBelief(
    const GameSimulator& env, float eps, std::vector<torch::Tensor> aoh, torch::Tensor seq_len, rela::TensorDict* retFeat) {
  auto feat = observe(env.state(), index, false, std::vector<int>(), std::vector<int>(), hideAction, true, true);
  seq_len[0] = int(aoh.size())+1;
  // auto feat = observe(env.state(), index, hideAction);
  if (retFeat != nullptr) {
    *retFeat = feat;
  }

  // forward bp regardless of whether rl is used
  {
    auto input = feat;
    addHid(input, bpHid_);
 
    assert(feat["s"].dim() == 1);

    // aoh should be empty on first call to observeBeforeActGeneralizedBelief
    auto aoh_complete = torch::zeros({67040});
    if (aoh.size() >= 1) {
      for (int j = 0; j < aoh.size(); ++j) {
        aoh_complete.slice(-1, 838*j, 838*(j+1)) = aoh[j].slice(-1, 0, 838);
      }
    }
    aoh_complete.slice(-1, 838*(seq_len[0].item<int>()-1), 838*seq_len[0].item<int>()) = feat["s"].slice(-1, 0, 838);
    input["aoh"] = aoh_complete;
    input["seq_len"] = seq_len;
  //  addHid(input, bpHid_);
    futBp_ = bpModel_->call("act", input);
    aoh.push_back(feat["s"]);
    return aoh;
  }

  // maybe forward rl
  if (rlStep_ > 0) {
    feat["eps"] = torch::tensor(std::vector<float>{eps});
    auto input = feat;
    addHid(input, rlHid_);
    futRl_ = rlModel_->call("act", input);
  }
  //auto a = aoh.accessor<float, 3>();
  //auto b = feat["s"].accessor<float, 1>();
  //for (int i = 0; i < 838; ++i) {
  //      auto x = b[i];
  //      a[seq_len[0].item<int>()-1][0][i] = x;
  //}
  return aoh;
}

int HybridModel::decideAction(
    const GameSimulator& env, bool verbose, rela::TensorDict* retAction) {
  // first get results from the futures, to update hid
  int action = -1;
  auto bpReply = futBp_.get();
  updateHid(bpReply, bpHid_);
  if (rlStep_ > 0) {
    auto rlReply = futRl_.get();
    updateHid(rlReply, rlHid_);
    action = rlReply.at("a").item<int64_t>();
    if (env.state().CurPlayer() == index) {
      --rlStep_;
    }

    if (verbose) {
      auto bpAction = bpReply.at("a").item<int64_t>();
      std::cout << "rl picks " << action << ", bp picks " << bpAction
                << ", remaining rl steps: " << rlStep_ << std::endl;
    }

    if (retAction != nullptr) {
      *retAction = rlReply;
    }
  } else {
    assert(futRl_.isNull());
    action = bpReply.at("a").item<int64_t>();
    if (verbose) {
      std::cout << "in act, action: " << action << std::endl;
    }

    // assert(retAction == nullptr);
    // technically this is not right, we should never return action from bp
    // for training purpose, but in this case it will be skip anyway.
    if (retAction != nullptr) {
      assert(action == env.game().MaxMoves());
      *retAction = bpReply;
    }
  }

  if (env.state().CurPlayer() != index) {
    assert(action == env.game().MaxMoves());
  }
  return action;
}
}  // namespace approx_search
