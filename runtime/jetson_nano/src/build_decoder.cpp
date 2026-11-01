// Copyright 2026 ZipVoice contributors
// SPDX-License-Identifier: Apache-2.0
// See the repository LICENSE for the full license text.

#include <NvInfer.h>
#include <NvOnnxParser.h>

#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
using namespace nvinfer1;
extern "C" void nano_register_relpos4d();
extern "C" IPluginV2* nano_make_relpos4d(int length);
extern "C" void nano_register_slice();
extern "C" IPluginV2* nano_make_slice(int K, int C, const int* idx);
extern "C" void nano_register_softmax();
extern "C" IPluginV2* nano_make_softmax();
extern "C" void nano_register_swoosh();
extern "C" IPluginV2* nano_make_swoosh(float c, float a, float b);
extern "C" void nano_register_fusions();
extern "C" IPluginV2* nano_make_attention_softmax();
extern "C" IPluginV2* nano_make_glu();
class Log : public ILogger {
  void log(Severity s, const char* text) noexcept override {
    std::fprintf(stderr, "[%d] %s\n", int(s), text);
  }
};
static void rewire(INetworkDefinition* network, ILayer* replacement,
                   ITensor* old) {
  auto* current = replacement->getOutput(0);
  for (int i = 0; i < network->getNbLayers(); i++) {
    auto* next = network->getLayer(i);
    if (next == replacement) continue;
    for (int j = 0; j < next->getNbInputs(); j++)
      if (next->getInput(j) == old) next->setInput(j, *current);
  }
}
int main(int argc, char** argv) {
  if (argc < 3) return 2;
  Log log;
  nano_register_fusions();
  nano_register_relpos4d();
  nano_register_slice();
  nano_register_softmax();
  nano_register_swoosh();
  auto* builder = createInferBuilder(log);
  auto* network = builder->createNetworkV2(0);
  auto* parser = nvonnxparser::createParser(*network, log);
  if (!parser->parseFromFile(argv[1], int(ILogger::Severity::kWARNING)))
    return 3;
  int softmax_count = 0;
  for (int i = 0; i < network->getNbLayers(); i++)
    if (network->getLayer(i)->getType() == LayerType::kSOFTMAX) {
      auto* soft = static_cast<ISoftMaxLayer*>(network->getLayer(i));
      if (soft->getAxes() != 1U ||
          soft->getInput(0)->getDimensions().nbDims != 2)
        throw std::runtime_error("Incorrect legacy Softmax shape/axis");
      ++softmax_count;
    }
  if (softmax_count != 16)
    throw std::runtime_error("Expected 16 corrected Softmax nodes");
  std::unordered_map<ITensor*, IConstantLayer*> constants;
  std::vector<IGatherLayer*> relpos;
  for (int i = 0; i < network->getNbLayers(); ++i) {
    auto* layer = network->getLayer(i);
    if (layer->getType() == LayerType::kCONSTANT)
      constants[layer->getOutput(0)] = static_cast<IConstantLayer*>(layer);
    if (layer->getType() == LayerType::kGATHER &&
        layer->getInput(0)->getDimensions().nbDims == 1 &&
        layer->getOutput(0)->getDimensions().nbDims == 1 &&
        layer->getOutput(0)->getDimensions().d[0] >= 262144)
      relpos.push_back(static_cast<IGatherLayer*>(layer));
  }
  std::vector<IPluginV2*> plugins;
  for (auto* layer : relpos) {
    int count = layer->getOutput(0)->getDimensions().d[0], length = 0;
    for (int l = 256; l <= 1024; l *= 2)
      if (count == 4 * l * l) length = l;
    if (!length || layer->getGatherAxis() != 0 ||
        layer->getInput(0)->getDimensions().d[0] !=
            4 * length * (2 * length - 1))
      throw std::runtime_error("Unexpected relative gather dimensions");
    auto it = constants.find(layer->getInput(1));
    if (it == constants.end())
      throw std::runtime_error("Gather indices are not constant");
    auto weights = it->second->getWeights();
    if (weights.type != DataType::kINT32 || weights.count != count)
      throw std::runtime_error("Unexpected index type");
    auto* indices = static_cast<const int*>(weights.values);
    for (int i = 0; i < count; i++) {
      int row = i / length;
      int expected =
          row * (2 * length - 1) + length - 1 - row % length + i % length;
      if (indices[i] != expected)
        throw std::runtime_error("Relative index formula mismatch");
    }
    auto* plugin = nano_make_relpos4d(length);
    plugins.push_back(plugin);
    std::unordered_map<ITensor*, ILayer*> producer;
    for (int i = 0; i < network->getNbLayers(); i++) {
      auto* l = network->getLayer(i);
      for (int j = 0; j < l->getNbOutputs(); j++) producer[l->getOutput(j)] = l;
    }
    ITensor* pre = layer->getInput(0);
    for (int s = 0; s < 4; s++) {
      auto* p = producer[pre];
      if (!p || p->getType() != LayerType::kSHUFFLE) break;
      pre = p->getInput(0);
    }
    auto pd = pre->getDimensions();
    if (pd.nbDims != 4 || pd.d[0] != 4 || pd.d[1] != 1 || pd.d[2] != length ||
        pd.d[3] != 2 * length - 1)
      throw std::runtime_error("Unexpected relpos pre-shape");
    ITensor* tail = layer->getOutput(0);
    for (int s = 0; s < 4; s++) {
      ILayer* nx = nullptr;
      for (int i = 0; i < network->getNbLayers(); i++) {
        auto* l = network->getLayer(i);
        if (l->getType() == LayerType::kSHUFFLE && l->getNbInputs() > 0 &&
            l->getInput(0) == tail) {
          nx = l;
          break;
        }
      }
      if (!nx) break;
      tail = nx->getOutput(0);
    }
    auto td = tail->getDimensions();
    if (td.nbDims != 4 || td.d[0] != 4 || td.d[1] != 1 || td.d[2] != length ||
        td.d[3] != length)
      throw std::runtime_error("Unexpected relpos tail shape");
    ITensor* inputs[] = {pre};
    auto* replacement = network->addPluginV2(inputs, 1, *plugin);
    replacement->setName(
        ("relative_copy4d_" + std::to_string(plugins.size())).c_str());
    rewire(network, replacement, tail);
    std::fprintf(stderr, "REPLACED_RELATIVE_CHAIN length=%d elements=%d\n",
                 length, count);
  }
  // Axis-2 constant-index Gathers are the model's channel slices (Slice lowered
  // as Gather). TensorRT 6's Gather costs ~6.8 ms each here; replace with a
  // strided copy.
  std::vector<IGatherLayer*> slices;
  for (int i = 0; i < network->getNbLayers(); ++i) {
    auto* layer = network->getLayer(i);
    if (layer->getType() != LayerType::kGATHER) continue;
    auto* g = static_cast<IGatherLayer*>(layer);
    auto inD = g->getInput(0)->getDimensions(),
         outD = g->getOutput(0)->getDimensions();
    if (g->getGatherAxis() != 2 || inD.nbDims != 3 || outD.nbDims != 3)
      continue;
    if (outD.d[0] != inD.d[0] || outD.d[1] != inD.d[1]) continue;
    int K = outD.d[2], C = inD.d[2];
    auto it = constants.find(g->getInput(1));
    if (it == constants.end()) continue;
    auto w = it->second->getWeights();
    if (w.type != DataType::kINT32 || w.count != K) continue;
    auto* idx = static_cast<const int*>(w.values);
    bool ok = true;
    for (int j = 0; j < K; j++)
      if (idx[j] < 0 || idx[j] >= C) {
        ok = false;
        break;
      }
    if (!ok) continue;
    slices.push_back(g);
  }
  for (auto* g : slices) {
    int K = g->getOutput(0)->getDimensions().d[2],
        C = g->getInput(0)->getDimensions().d[2];
    auto w = constants[g->getInput(1)]->getWeights();
    auto* idx = static_cast<const int*>(w.values);
    auto* plugin = nano_make_slice(K, C, idx);
    plugins.push_back(plugin);
    ITensor* inputs[] = {g->getInput(0)};
    auto* replacement = network->addPluginV2(inputs, 1, *plugin);
    replacement->setName(
        ("slice_copy_" + std::to_string(plugins.size())).c_str());
    rewire(network, replacement, g->getOutput(0));
  }
  std::fprintf(stderr,
               "REPLACED_SLICE_GATHERS=%zu relpos=%zu total_plugins=%zu\n",
               slices.size(), relpos.size(), plugins.size());
  // Replace the whole row-softmax chain: [4,1,Q,K] -> shuffle -> softmax ->
  // shuffle -> [4,1,Q,K]
  std::vector<ISoftMaxLayer*> sms;
  for (int i = 0; i < network->getNbLayers(); i++)
    if (network->getLayer(i)->getType() == LayerType::kSOFTMAX)
      sms.push_back(static_cast<ISoftMaxLayer*>(network->getLayer(i)));
  std::unordered_map<ITensor*, ILayer*> prod2;
  for (int i = 0; i < network->getNbLayers(); i++) {
    auto* l = network->getLayer(i);
    for (int j = 0; j < l->getNbOutputs(); j++) prod2[l->getOutput(j)] = l;
  }
  int sm_replaced = 0;
  for (auto* sm : sms) {
    ITensor* pre = sm->getInput(0);
    for (int s = 0; s < 3; s++) {
      auto* p = prod2[pre];
      if (!p || p->getType() != LayerType::kSHUFFLE) break;
      pre = p->getInput(0);
    }
    auto pd = pre->getDimensions();
    if (pd.nbDims < 2) continue;
    ITensor* tail = sm->getOutput(0);
    for (int s = 0; s < 3; s++) {
      ILayer* nx = nullptr;
      for (int i = 0; i < network->getNbLayers(); i++) {
        auto* l = network->getLayer(i);
        if (l->getType() == LayerType::kSHUFFLE && l->getNbInputs() > 0 &&
            l->getInput(0) == tail) {
          nx = l;
          break;
        }
      }
      if (!nx) break;
      tail = nx->getOutput(0);
    }
    auto td = tail->getDimensions();
    if (td.nbDims != pd.nbDims) continue;
    bool same = true;
    for (int k = 0; k < pd.nbDims; k++)
      if (pd.d[k] != td.d[k]) same = false;
    if (!same) continue;
    auto* plugin = nano_make_softmax();
    plugins.push_back(plugin);
    ITensor* inputs[] = {pre};
    auto* replacement = network->addPluginV2(inputs, 1, *plugin);
    replacement->setName(
        ("row_softmax_" + std::to_string(plugins.size())).c_str());
    rewire(network, replacement, tail);
    sm_replaced++;
  }
  std::fprintf(stderr, "REPLACED_SOFTMAX_CHAINS=%d of %zu\n", sm_replaced,
               sms.size());
  int act_total = 0, soft_total = 0;
  for (int i = 0; i < network->getNbLayers(); i++) {
    auto* l = network->getLayer(i);
    if (l->getType() != LayerType::kACTIVATION) continue;
    act_total++;
    auto* a = static_cast<IActivationLayer*>(l);
    if (a->getActivationType() == ActivationType::kSOFTPLUS) soft_total++;
  }
  std::fprintf(stderr, "ACTIVATIONS=%d SOFTPLUS=%d\n", act_total, soft_total);
  // Fuse Swoosh: softplus(x-c) - a*x - b
  std::unordered_map<ITensor*, ILayer*> prod4;
  for (int i = 0; i < network->getNbLayers(); i++) {
    auto* l = network->getLayer(i);
    for (int j = 0; j < l->getNbOutputs(); j++) prod4[l->getOutput(j)] = l;
  }
  auto scalar = [&](ITensor* t, float& v) -> bool {
    ILayer* l = prod4[t];
    if (!l || l->getType() != LayerType::kCONSTANT) return false;
    auto w = static_cast<IConstantLayer*>(l)->getWeights();
    if (w.type != DataType::kFLOAT || w.count != 1) return false;
    v = static_cast<const float*>(w.values)[0];
    return true;
  };
  int sw = 0;
  for (int i = 0; i < network->getNbLayers(); i++) {
    auto* l = network->getLayer(i);
    if (l->getType() != LayerType::kACTIVATION) continue;
    auto* act = static_cast<IActivationLayer*>(l);
    if (act->getActivationType() != ActivationType::kSOFTPLUS) continue;
    {
      static int dbg = 0;
      if (dbg < 3) {
        ILayer* p = prod4[act->getInput(0)];
        int pt = p ? (int)p->getType() : -1;
        int op = (p && p->getType() == LayerType::kELEMENTWISE)
                     ? (int)static_cast<IElementWiseLayer*>(p)->getOperation()
                     : -9;
        float cc = 0;
        bool okc =
            p && p->getType() == LayerType::kELEMENTWISE
                ? scalar(static_cast<IElementWiseLayer*>(p)->getInput(1), cc)
                : false;
        std::fprintf(stderr,
                     "SWDBG prod_type=%d prod_op=%d scalar_ok=%d c=%g\n", pt,
                     op, int(okc), cc);
        if (dbg == 0) {
          for (int q = 0; q < network->getNbLayers(); q++) {
            auto* ll = network->getLayer(q);
            if (ll->getType() != LayerType::kELEMENTWISE) continue;
            auto* ee = static_cast<IElementWiseLayer*>(ll);
            for (int a = 0; a < 2; a++)
              if (ee->getInput(a) == act->getOutput(0)) {
                ITensor* o = ee->getInput(1 - a);
                ILayer* pp = prod4[o];
                std::fprintf(stderr,
                             "SWDBG consumer op=%d other_prod_type=%d "
                             "other_prod_op=%d\n",
                             (int)ee->getOperation(),
                             pp ? (int)pp->getType() : -1,
                             (pp && pp->getType() == LayerType::kELEMENTWISE)
                                 ? (int)static_cast<IElementWiseLayer*>(pp)
                                       ->getOperation()
                                 : -9);
              }
          }
        }
        dbg++;
      }
    }
    ILayer* sub4 = prod4[act->getInput(0)];
    if (!sub4 || sub4->getType() != LayerType::kELEMENTWISE) continue;
    auto* ew4 = static_cast<IElementWiseLayer*>(sub4);
    if (ew4->getOperation() != ElementWiseOperation::kSUB) continue;
    float c;
    if (!scalar(ew4->getInput(1), c)) continue;
    ITensor* x = ew4->getInput(0);
    auto scale_val = [&](ILayer* m, float& v) -> bool {
      if (!m || m->getType() != LayerType::kSCALE) return false;
      auto* sl = static_cast<IScaleLayer*>(m);
      if (sl->getMode() != ScaleMode::kUNIFORM) return false;
      auto w = sl->getScale();
      if (w.type != DataType::kFLOAT || w.count != 1) return false;
      v = static_cast<const float*>(w.values)[0];
      return true;
    };
    auto shift_val = [&](ILayer* m, float& v) -> bool {
      if (!m || m->getType() != LayerType::kSCALE) return false;
      auto* sl = static_cast<IScaleLayer*>(m);
      if (sl->getMode() != ScaleMode::kUNIFORM) return false;
      auto w = sl->getShift();
      if (w.type != DataType::kFLOAT || w.count != 1) return false;
      v = static_cast<const float*>(w.values)[0];
      return true;
    };
    ILayer* s1 = nullptr;
    float av = 0;
    for (int j = 0; j < network->getNbLayers() && !s1; j++) {
      auto* q = network->getLayer(j);
      if (q->getType() != LayerType::kELEMENTWISE) continue;
      auto* e = static_cast<IElementWiseLayer*>(q);
      if (e->getOperation() != ElementWiseOperation::kSUB) continue;
      for (int a = 0; a < 2; a++) {
        if (e->getInput(a) != act->getOutput(0)) continue;
        ILayer* m = prod4[e->getInput(1 - a)];
        float tmp = 0;
        if (scale_val(m, tmp) && m->getInput(0) == x) {
          s1 = q;
          av = tmp;
          break;
        }
      }
    }
    if (!s1) continue;
    ILayer* s2 = nullptr;
    float bv = 0;
    for (int j = 0; j < network->getNbLayers() && !s2; j++) {
      auto* q = network->getLayer(j);
      if (q->getType() == LayerType::kSCALE) {
        auto* sl = static_cast<IScaleLayer*>(q);
        if (sl->getInput(0) == s1->getOutput(0)) {
          float sh;
          if (shift_val(q, sh)) {
            s2 = q;
            bv = -sh;
            break;
          }
        }
        continue;
      }
      if (q->getType() != LayerType::kELEMENTWISE) continue;
      auto* e = static_cast<IElementWiseLayer*>(q);
      if (e->getOperation() != ElementWiseOperation::kSUB) continue;
      for (int a = 0; a < 2; a++) {
        if (e->getInput(a) != s1->getOutput(0)) continue;
        if (scalar(e->getInput(1 - a), bv)) {
          s2 = q;
          break;
        }
      }
    }
    if (!s2) continue;
    ITensor* tail = s2->getOutput(0);
    auto* plugin = nano_make_swoosh(c, av, bv);
    plugins.push_back(plugin);
    ITensor* ins[] = {x};
    auto* rep = network->addPluginV2(ins, 1, *plugin);
    rep->setName(("swoosh_" + std::to_string(plugins.size())).c_str());
    rewire(network, rep, tail);
    sw++;
  }
  std::fprintf(stderr, "REPLACED_SWOOSH=%d\n", sw);

  // Fuse only exact validated graph patterns, after existing compatibility
  // rewrites.
  std::unordered_map<ITensor*, ILayer*> producer;
  for (int i = 0; i < network->getNbLayers(); i++) {
    auto* l = network->getLayer(i);
    for (int j = 0; j < l->getNbOutputs(); j++) producer[l->getOutput(j)] = l;
  }
  auto plugin_type = [](ILayer* l, const char* name) {
    return l && l->getType() == LayerType::kPLUGIN_V2 &&
           std::string(
               static_cast<IPluginV2Layer*>(l)->getPlugin().getPluginType()) ==
               name;
  };
  int fused_attention = 0, fused_glu = 0;
  int original_layers = network->getNbLayers();
  for (int i = 0; i < original_layers; i++) {
    auto* sm = network->getLayer(i);
    if (!plugin_type(sm, "NanoRowSoftmax")) continue;
    auto it = producer.find(sm->getInput(0));
    if (it == producer.end()) continue;
    auto* sum = it->second;
    if (sum->getType() != LayerType::kELEMENTWISE ||
        static_cast<IElementWiseLayer*>(sum)->getOperation() !=
            ElementWiseOperation::kSUM)
      continue;
    for (int side = 0; side < 2; side++) {
      auto it2 = producer.find(sum->getInput(side));
      if (it2 == producer.end() || !plugin_type(it2->second, "NanoRelPos4D"))
        continue;
      auto* rp = it2->second;
      auto* qk = sum->getInput(1 - side);
      auto* relative = rp->getInput(0);
      auto d = qk->getDimensions(), rd = relative->getDimensions();
      if (d.nbDims != 4 || rd.nbDims != 4 || d.d[0] != 4 || d.d[1] != 1 ||
          d.d[2] != d.d[3] || rd.d[0] != 4 || rd.d[1] != 1 ||
          rd.d[2] != d.d[2] || rd.d[3] != 2 * d.d[3] - 1)
        throw std::runtime_error("fused attention shape mismatch");
      auto* p = nano_make_attention_softmax();
      plugins.push_back(p);
      ITensor* ins[] = {qk, relative};
      auto* rep = network->addPluginV2(ins, 2, *p);
      rep->setName(
          ("fused_attention_" + std::to_string(fused_attention++)).c_str());
      rewire(network, rep, sm->getOutput(0));
      break;
    }
  }
  auto slice_start = [&](ILayer* l, int& K, int& C, int& first) -> bool {
    if (!plugin_type(l, "NanoSliceCopy")) return false;
    auto& p = static_cast<IPluginV2Layer*>(l)->getPlugin();
    std::vector<int> v((p.getSerializationSize() + 3) / 4);
    p.serialize(v.data());
    K = v[0];
    C = v[2];
    if (K < 1 || C != 2 * K || v.size() != size_t(K + 4)) return false;
    first = v[4];
    for (int j = 0; j < K; j++)
      if (v[j + 4] != first + j) return false;
    return true;
  };
  for (int i = 0; i < original_layers; i++) {
    auto* l = network->getLayer(i);
    if (l->getType() != LayerType::kELEMENTWISE ||
        static_cast<IElementWiseLayer*>(l)->getOperation() !=
            ElementWiseOperation::kPROD)
      continue;
    for (int side = 0; side < 2; side++) {
      auto it = producer.find(l->getInput(side));
      if (it == producer.end() ||
          it->second->getType() != LayerType::kACTIVATION)
        continue;
      auto* sig = static_cast<IActivationLayer*>(it->second);
      if (sig->getActivationType() != ActivationType::kSIGMOID) continue;
      auto ia = producer.find(l->getInput(1 - side)),
           ib = producer.find(sig->getInput(0));
      if (ia == producer.end() || ib == producer.end()) continue;
      int ka, ca, fa, kb, cb, fb;
      if (!slice_start(ia->second, ka, ca, fa) ||
          !slice_start(ib->second, kb, cb, fb) || ka != kb || ca != cb ||
          fa != 0 || fb != ka ||
          ia->second->getInput(0) != ib->second->getInput(0))
        continue;
      auto* p = nano_make_glu();
      plugins.push_back(p);
      ITensor* ins[] = {ia->second->getInput(0)};
      auto* rep = network->addPluginV2(ins, 1, *p);
      rep->setName(("fused_glu_" + std::to_string(fused_glu++)).c_str());
      rewire(network, rep, l->getOutput(0));
      break;
    }
  }
  std::fprintf(stderr, "FUSED_ATTENTION=%d FUSED_GLU=%d\n", fused_attention,
               fused_glu);
  if (fused_attention != 16 || fused_glu != 32)
    throw std::runtime_error("fusion coverage mismatch");

  if (argc > 3 && std::string(argv[3]) == "dump") {
    std::printf("TOTAL_LAYERS %d\n", network->getNbLayers());
    for (int i = 0; i < network->getNbLayers(); i++) {
      auto* l = network->getLayer(i);
      const char* t = "Other";
      switch (l->getType()) {
        case LayerType::kCONVOLUTION:
          t = "Conv";
          break;
        case LayerType::kMATRIX_MULTIPLY:
          t = "MatMul";
          break;
        case LayerType::kSHUFFLE:
          t = "Shuffle";
          break;
        case LayerType::kSOFTMAX:
          t = "Softmax";
          break;
        case LayerType::kELEMENTWISE:
          t = "ElementWise";
          break;
        case LayerType::kACTIVATION:
          t = "Activation";
          break;
        case LayerType::kCONSTANT:
          t = "Constant";
          break;
        case LayerType::kGATHER:
          t = "Gather";
          break;
        case LayerType::kREDUCE:
          t = "Reduce";
          break;
        case LayerType::kUNARY:
          t = "Unary";
          break;
        case LayerType::kPLUGIN_V2:
          t = "Plugin";
          break;
        case LayerType::kIDENTITY:
          t = "Identity";
          break;
        case LayerType::kSCALE:
          t = "Scale";
          break;
        case LayerType::kCONCATENATION:
          t = "Concat";
          break;
        case LayerType::kSLICE:
          t = "Slice";
          break;
        default:
          break;
      }
      std::printf("L%03d %-11s %-48s out=", i, t, l->getName());
      for (int j = 0; j < l->getNbOutputs(); j++) {
        auto d = l->getOutput(j)->getDimensions();
        std::printf("[");
        for (int k = 0; k < d.nbDims; k++)
          std::printf("%d%s", d.d[k], k + 1 < d.nbDims ? "," : "");
        std::printf("]");
      }
      std::printf(" in=");
      if (l->getNbInputs() > 0) {
        auto d = l->getInput(0)->getDimensions();
        std::printf("[");
        for (int k = 0; k < d.nbDims; k++)
          std::printf("%d%s", d.d[k], k + 1 < d.nbDims ? "," : "");
        std::printf("]");
      }
      std::printf("\n");
    }
    return 0;
  }
  builder->setMaxBatchSize(1);
  builder->setMaxWorkspaceSize(64ULL << 20);
  builder->setFp16Mode(true);
  builder->setMinFindIterations(1);
  builder->setAverageFindIterations(1);
  auto* engine = builder->buildCudaEngine(*network);
  if (!engine) return 4;
  auto* blob = engine->serialize();
  std::ofstream out(argv[2], std::ios::binary);
  out.write(static_cast<const char*>(blob->data()), blob->size());
  out.close();
  if (!out) return 5;
  std::fprintf(stderr, "NATIVE_BUILD_PASSED bytes=%zu\n", blob->size());
  blob->destroy();
  engine->destroy();
  parser->destroy();
  network->destroy();
  builder->destroy();
  return 0;
}
