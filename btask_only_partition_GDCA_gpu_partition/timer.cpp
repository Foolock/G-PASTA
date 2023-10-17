#include <ot/timer/timer.hpp>

namespace ot {

// ------------------------------------------------------------------------------------------------

// Function: set_num_threads
Timer& Timer::set_num_threads(unsigned n) {
  std::scoped_lock lock(_mutex);
  unsigned w = (n == 0) ? 0 : n-1;
  OT_LOGI("using ", n, " threads (", w, " worker)");
  // TODO
  //_taskflow.num_workers(w);
  return *this;
}

// Procedure: _add_to_lineage
void Timer::_add_to_lineage(tf::Task task) {
  _lineage | [&] (auto& p) { p.precede(task); };
  _lineage = task;
}

// Function: _max_pin_name_size
size_t Timer::_max_pin_name_size() const {
  if(_pins.empty()) {
    return 0;
  }
  else {
    return std::max_element(_pins.begin(), _pins.end(), 
      [] (const auto& l, const auto& r) {
        return l.second._name.size() < r.second._name.size();
      }
    )->second._name.size();
  }
}

// Function: _max_net_name_size
size_t Timer::_max_net_name_size() const {
  if(_nets.empty()) {
    return 0;
  }
  else {
    return std::max_element(_nets.begin(), _nets.end(), 
      [] (const auto& l, const auto& r) {
        return l.second._name.size() < r.second._name.size();
      }
    )->second._name.size();
  }
}

// Function: repower_gate
// Change the size or level of an existing gate, e.g., NAND2_X2 to NAND2_X3. The gate's
// logic function and topology is guaranteed to be the same, along with the currently-connected
// nets. However, the pin capacitances of the new cell type might be different. 
Timer& Timer::repower_gate(std::string gate, std::string cell) {

  std::scoped_lock lock(_mutex);

  auto task = _taskflow.emplace([this, gate=std::move(gate), cell=std::move(cell)] () {
    _repower_gate(gate, cell);
  });
  
  _add_to_lineage(task);

  return *this;
}

// Procedure: _repower_gate
void Timer::_repower_gate(const std::string& gname, const std::string& cname) {
  
  OT_LOGE_RIF(!_celllib[MIN] || !_celllib[MAX], "celllib not found");

  // Insert the gate if it doesn't exist.
  if(auto gitr = _gates.find(gname); gitr == _gates.end()) {
    OT_LOGW("gate ", gname, " doesn't exist (insert instead)");
    _insert_gate(gname, cname);
    return;
  }
  else {

    auto cell = CellView {_celllib[MIN]->cell(cname), _celllib[MAX]->cell(cname)};

    OT_LOGE_RIF(!cell[MIN] || !cell[MAX], "cell ", cname, " not found");

    auto& gate = gitr->second;

    // Remap the cellpin
    for(auto pin : gate._pins) {
      FOR_EACH_EL(el) {
        assert(pin->cellpin(el));
        if(const auto cpin = cell[el]->cellpin(pin->cellpin(el)->name)) {
          pin->_remap_cellpin(el, *cpin);
        }
        else {
          OT_LOGE(
            "repower ", gname, " with ", cname, " failed (cellpin mismatched)"
          );  
        }
      }
    }
    
    gate._cell = cell;

    // reconstruct the timing and tests
    _remove_gate_arcs(gate);
    _insert_gate_arcs(gate);

    // Insert the gate to the frontier
    for(auto pin : gate._pins) {
      _insert_frontier(*pin);
      for(auto arc : pin->_fanin) {
        _insert_frontier(arc->_from);
      }
    }
  }
}

// Fucntion: insert_gate
// Create a new gate in the design. This newly-created gate is "not yet" connected to
// any other gates or wires. The gate to insert cannot conflict with existing gates.
Timer& Timer::insert_gate(std::string gate, std::string cell) {  
  
  std::scoped_lock lock(_mutex);

  auto op = _taskflow.emplace([this, gate=std::move(gate), cell=std::move(cell)] () {
    _insert_gate(gate, cell);
  });

  _add_to_lineage(op);

  return *this;
}

// Function: _insert_gate
void Timer::_insert_gate(const std::string& gname, const std::string& cname) {

  OT_LOGE_RIF(!_celllib[MIN] || !_celllib[MAX], "celllib not found");

  if(_gates.find(gname) != _gates.end()) {
    OT_LOGW("gate ", gname, " already existed");
    return;
  }

  auto cell = CellView {_celllib[MIN]->cell(cname), _celllib[MAX]->cell(cname)};

  if(!cell[MIN] || !cell[MAX]) {
    OT_LOGE("cell ", cname, " not found in celllib");
    return;
  }
  
  auto& gate = _gates.try_emplace(gname, gname, cell).first->second;
  
  // Insert pins
  for(const auto& [cpname, ecpin] : cell[MIN]->cellpins) {

    CellpinView cpv {&ecpin, cell[MAX]->cellpin(cpname)};

    if(!cpv[MIN] || !cpv[MAX]) {
      OT_LOGF("cellpin ", cpname, " mismatched in celllib");
    }

    auto& pin = _insert_pin(gname + ':' + cpname);
    pin._handle = cpv;
    pin._gate = &gate;
    
    gate._pins.push_back(&pin);
  }
  
  _insert_gate_arcs(gate);
}

// Fucntion: remove_gate
// Remove a gate from the current design. This is guaranteed to be called after the gate has 
// been disconnected from the design using pin-level operations. The procedure iterates all 
// pins in the cell to which the gate was attached. Each pin that is being iterated is either
// a cell input pin or cell output pin. In the former case, the pin might have constraint arc
// while in the later case, the ot_pin.has no output connections and all fanin edges should be 
// removed here.
Timer& Timer::remove_gate(std::string gate) {  
  
  std::scoped_lock lock(_mutex);

  auto op = _taskflow.emplace([this, gate=std::move(gate)] () {
    if(auto gitr = _gates.find(gate); gitr != _gates.end()) {
      _remove_gate(gitr->second);
    }
  });

  _add_to_lineage(op);

  return *this;
}

// Procedure: _remove_gate
void Timer::_remove_gate(Gate& gate) {

  // Disconnect this gate from the design.
  for(auto pin : gate._pins) {
    _disconnect_pin(*pin);
  }

  // Remove associated test
  for(auto test : gate._tests) {
    _remove_test(*test);
  }

  // Remove associated arcs
  for(auto arc : gate._arcs) {
    _remove_arc(*arc);
  }

  // Disconnect the gate and remove the pins from the gate
  for(auto pin : gate._pins) {
    _remove_pin(*pin);
  }

  // remove the gate
  _gates.erase(gate._name);
}

// Procedure: _remove_gate_arcs
void Timer::_remove_gate_arcs(Gate& gate) {

  // remove associated tests
  for(auto test : gate._tests) {
    _remove_test(*test);
  }
  gate._tests.clear();
  
  // remove associated arcs
  for(auto arc : gate._arcs) {
    _remove_arc(*arc);
  }
  gate._arcs.clear();
}

// Procedure: _insert_gate_arcs
void Timer::_insert_gate_arcs(Gate& gate) {

  assert(gate._tests.empty() && gate._arcs.empty());

  FOR_EACH_EL(el) {
    for(const auto& [cpname, cp] : gate._cell[el]->cellpins) {
      auto& to_pin = _insert_pin(gate._name + ':' + cpname);

      for(const auto& tm : cp.timings) {

        if(_is_redundant_timing(tm, el)) {
          continue;
        }

        TimingView tv{nullptr, nullptr};
        tv[el] = &tm;

        auto& from_pin = _insert_pin(gate._name + ':' + tm.related_pin);
        auto& arc = _insert_arc(from_pin, to_pin, tv);
        
        gate._arcs.push_back(&arc);
        if(tm.is_constraint()) {
          auto& test = _insert_test(arc);
          gate._tests.push_back(&test);
        }
      }
    }
  }
}

// Function: connect_pin
// Connect the pin to the corresponding net. The pin_name will either have the 
// <gate name>:<cell pin name> syntax (e.g., u4:ZN) or be a primary input. The net name
// will match an existing net read in from a .spef file.
Timer& Timer::connect_pin(std::string pin, std::string net) {

  std::scoped_lock lock(_mutex);

  auto op = _taskflow.emplace([this, pin=std::move(pin), net=std::move(net)] () {
    auto p = _pins.find(pin);
    auto n = _nets.find(net);
    OT_LOGE_RIF(p==_pins.end() || n == _nets.end(),
      "can't connect pin ", pin,  " to net ", net, " (pin/net not found)"
    )
    _connect_pin(p->second, n->second);
  });

  _add_to_lineage(op);

  return *this;
}

// Procedure: _connect_pin
void Timer::_connect_pin(Pin& pin, Net& net) {
      
  // Connect the pin to the net and construct the edge connections.
  net._insert_pin(pin);
  
  // Case 1: the pin is the root of the net.
  if(&pin == net._root) {
    for(auto leaf : net._pins) {
      if(leaf != &pin) {
        _insert_arc(pin, *leaf, net);
      }
    }
  }
  // Case 2: the pin is not a root of the net.
  else {
    if(net._root) {
      _insert_arc(*net._root, pin, net);
    }
  }

  // TODO(twhuang) Enable the clock tree update?
}

// Procedure: disconnect_pin
// Disconnect the pin from the net it is connected to. The pin_name will either have the 
// <gate name>:<cell pin name> syntax (e.g., u4:ZN) or be a primary input.
Timer& Timer::disconnect_pin(std::string name) {
  
  std::scoped_lock lock(_mutex);

  auto op = _taskflow.emplace([this, name=std::move(name)] () {
    if(auto itr = _pins.find(name); itr != _pins.end()) {
      _disconnect_pin(itr->second);
    }
  });

  _add_to_lineage(op);

  return *this;
}

// Procedure: disconnect_pin
// TODO (twhuang)
// try get rid of find_fanin which can be wrong under multiple arcs.
void Timer::_disconnect_pin(Pin& pin) {

  auto net = pin._net;

  if(net == nullptr) return;

  // Case 1: the pin is a root of the net (i.e., root of the rctree)
  if(&pin == net->_root) {
    // Iterate the pinlist and delete the corresponding edge. Notice here we cannot iterate
    // fanout of the node during removal.
    for(auto leaf : net->_pins) {
      if(leaf != net->_root) {
        auto arc = leaf->_find_fanin(*net->_root);
        assert(arc);
        _remove_arc(*arc);
      }
    }
  }
  // Case 2: the pin is not a root of the net.
  else {
    if(net->_root) {
      auto arc = pin._find_fanin(*net->_root);
      assert(arc);
      _remove_arc(*arc);
    }
  }
  
  // TODO: Enable the clock tree update.
  
  // Remove the pin from the net and enable the rc timing update.
  net->_remove_pin(pin);
}

// Function: insert_net
// Creates an empty net object with the input "net_name". By default, it will not be connected 
// to any pins and have no parasitics (.spef). This net will be connected to existing pins in 
// the design by the "connect_pin" and parasitics will be loaded by "spef".
Timer& Timer::insert_net(std::string name) {

  std::scoped_lock lock(_mutex);

  auto op = _taskflow.emplace([this, name=std::move(name)] () {
    _insert_net(name);
  });

  _add_to_lineage(op);

  return *this;
}

// Function: _insert_net
Net& Timer::_insert_net(const std::string& name) {
  return _nets.try_emplace(name, name).first->second;
}

// Procedure: remove_net
// Remove a net from the current design, which by default removes all associated pins.
Timer& Timer::remove_net(std::string name) {

  std::scoped_lock lock(_mutex);

  auto op = _taskflow.emplace([this, name=std::move(name)] () {
    if(auto itr = _nets.find(name); itr != _nets.end()) {
      _remove_net(itr->second);
    }
  });

  _add_to_lineage(op);

  return *this;
}

// Function: _remove_net
void Timer::_remove_net(Net& net) {

  if(net.num_pins() > 0) {
    auto fetch = net._pins;
    for(auto pin : fetch) {
      _disconnect_pin(*pin);
    }
  }

  _nets.erase(net._name);
}

// Function: _insert_pin
Pin& Timer::_insert_pin(const std::string& name) {
  
  // pin already exists
  if(auto [itr, inserted] = _pins.try_emplace(name, name); !inserted) {
    return itr->second;
  }
  // inserted a new pon
  else {
    
    // Generate the pin idx
    auto& pin = itr->second;
    
    // Assign the idx mapping
    pin._idx = _pin_idx_gen.get();
    resize_to_fit(pin._idx + 1, _idx2pin);
    _idx2pin[pin._idx] = &pin;

    // insert to frontier
    _insert_frontier(pin);

    return pin;
  }
}

// Function: _remove_pin
void Timer::_remove_pin(Pin& pin) {

  assert(pin.num_fanouts() == 0 && pin.num_fanins() == 0 && pin.net() == nullptr);

  _remove_frontier(pin);

  // remove the id mapping
  _idx2pin[pin._idx] = nullptr;
  _pin_idx_gen.recycle(pin._idx);

  // remove the pin
  _pins.erase(pin._name);
}

// Function: cppr
Timer& Timer::cppr(bool flag) {
  
  std::scoped_lock lock(_mutex);

  auto op = _taskflow.emplace([this, flag] () {
    _cppr(flag);
  });

  _add_to_lineage(op);

  return *this;
}

// Procedure: _cppr
// Enable/Disable common path pessimism removal (cppr) analysis
void Timer::_cppr(bool enable) {
  
  // nothing to do.
  if((enable && _cppr_analysis) || (!enable && !_cppr_analysis)) {
    return;
  }

  if(enable) {
    OT_LOGI("enable cppr analysis");
    _cppr_analysis.emplace();
  }
  else {
    OT_LOGI("disable cppr analysis");
    _cppr_analysis.reset();
  }
    
  for(auto& test : _tests) {
    _insert_frontier(test._constrained_pin());
  }
}

// Function: clock
Timer& Timer::create_clock(std::string c, std::string s, float p) {
  
  std::scoped_lock lock(_mutex);

  auto op = _taskflow.emplace([this, c=std::move(c), s=std::move(s), p] () {
    if(auto itr = _pins.find(s); itr != _pins.end()) {
      _create_clock(c, itr->second, p);
    }
    else {
      OT_LOGE("can't create clock ", c, " on source ", s, " (pin not found)");
    }
  });

  _add_to_lineage(op);
  
  return *this;
}

// Function: create_clock
Timer& Timer::create_clock(std::string c, float p) {
  
  std::scoped_lock lock(_mutex);

  auto op = _taskflow.emplace([this, c=std::move(c), p] () {
    _create_clock(c, p);
  });

  _add_to_lineage(op);

  return *this;
}

// Procedure: _create_clock
Clock& Timer::_create_clock(const std::string& name, Pin& pin, float period) {
  auto& clock = _clocks.try_emplace(name, name, pin, period).first->second;
  _insert_frontier(pin);
  return clock;
}

// Procedure: _create_clock
Clock& Timer::_create_clock(const std::string& name, float period) {
  auto& clock = _clocks.try_emplace(name, name, period).first->second;
  return clock;
}

// Function: insert_primary_input
Timer& Timer::insert_primary_input(std::string name) {

  std::scoped_lock lock(_mutex);

  auto op = _taskflow.emplace([this, name=std::move(name)] () {
    _insert_primary_input(name);
  });

  _add_to_lineage(op);

  return *this;
}

// Procedure: _insert_primary_input
void Timer::_insert_primary_input(const std::string& name) {

  if(_pis.find(name) != _pis.end()) {
    OT_LOGW("can't insert PI ", name, " (already existed)");
    return;
  }

  assert(_pins.find(name) == _pins.end());

  // Insert the pin and and pi
  auto& pin = _insert_pin(name);
  auto& pi = _pis.try_emplace(name, pin).first->second;
  
  // Associate the connection.
  pin._handle = &pi;

  // Insert the pin to the frontier
  _insert_frontier(pin);

  // Create a net for the po and connect the pin to the net.
  auto& net = _insert_net(name); 
  
  // Connect the pin to the net.
  _connect_pin(pin, net);
}

// Function: insert_primary_output
Timer& Timer::insert_primary_output(std::string name) {

  std::scoped_lock lock(_mutex);

  auto op = _taskflow.emplace([this, name=std::move(name)] () {
    _insert_primary_output(name);
  });

  _add_to_lineage(op);

  return *this;
}

// Procedure: _insert_primary_output
void Timer::_insert_primary_output(const std::string& name) {

  if(_pos.find(name) != _pos.end()) {
    OT_LOGW("can't insert PO ", name, " (already existed)");
    return;
  }

  assert(_pins.find(name) == _pins.end());

  // Insert the pin and and pi
  auto& pin = _insert_pin(name);
  auto& po = _pos.try_emplace(name, pin).first->second;
  
  // Associate the connection.
  pin._handle = &po;

  // Insert the pin to the frontier
  _insert_frontier(pin);

  // Create a net for the po and connect the pin to the net.
  auto& net = _insert_net(name); 

  // Connect the pin to the net.
  _connect_pin(pin, net);
}

// Procedure: _insert_test
Test& Timer::_insert_test(Arc& arc) {
  auto& test = _tests.emplace_front(arc);
  test._satellite = _tests.begin();
  test._pin_satellite = arc._to._tests.insert(arc._to._tests.end(), &test);
  return test;
}

// Procedure: _remove_test
void Timer::_remove_test(Test& test) {
  assert(test._satellite);
  if(test._pin_satellite) {
    test._arc._to._tests.erase(*test._pin_satellite);
  }
  _tests.erase(*test._satellite);
}

// Procedure: _remove_arc
// Remove an arc from the design. The procedure first disconnects the arc from its two ending
// pins, "from_pin" and "to_pin". Then it removes the arc from the design and insert both
// "from_pin" and "to_pin" into the pipeline.
void Timer::_remove_arc(Arc& arc) {

  assert(arc._satellite);
  
  arc._from._remove_fanout(arc);
  arc._to._remove_fanin(arc);

  // Insert the two ends to the frontier list.
  _insert_frontier(arc._from, arc._to);
  
  // remove the id mapping
  _idx2arc[arc._idx] = nullptr;
  _arc_idx_gen.recycle(arc._idx);

  // Remove this arc from the timer.
  _arcs.erase(*arc._satellite);
}

// Function: _insert_arc (net arc)
// Insert an net arc to the timer.
Arc& Timer::_insert_arc(Pin& from, Pin& to, Net& net) {

  OT_LOGF_IF(&from == &to, "net arc is a self loop at ", to._name);

  // Create a new arc
  auto& arc = _arcs.emplace_front(from, to, net);
  arc._satellite = _arcs.begin();

  from._insert_fanout(arc);
  to._insert_fanin(arc);

  // Insert frontiers
  _insert_frontier(from, to);
   
  // Assign the idx mapping
  arc._idx = _arc_idx_gen.get();
  resize_to_fit(arc._idx + 1, _idx2arc);
  _idx2arc[arc._idx] = &arc;

  return arc;
}

// Function: _insert_arc (cell arc)
// Insert a cell arc to the timing graph. A cell arc is a combinational link.
Arc& Timer::_insert_arc(Pin& from, Pin& to, TimingView tv) {
  
  //OT_LOGF_IF(&from == &to, "timing graph contains a self loop at ", to._name);

  // Create a new arc
  auto& arc = _arcs.emplace_front(from, to, tv);
  arc._satellite = _arcs.begin();
  from._insert_fanout(arc);
  to._insert_fanin(arc);

  // insert the arc into frontier list.
  _insert_frontier(from, to);
  
  // Assign the idx mapping
  arc._idx = _arc_idx_gen.get();
  resize_to_fit(arc._idx + 1, _idx2arc);
  _idx2arc[arc._idx] = &arc;

  return arc;
}

// Procedure: _fprop_rc_timing
void Timer::_fprop_rc_timing(Pin& pin) {
  if(auto net = pin._net; net) {
    net->_update_rc_timing();
  }
}

// Procedure: _fprop_slew
void Timer::_fprop_slew(Pin& pin) {
  
  // clear slew  
  pin._reset_slew();

  // PI
  if(auto pi = pin.primary_input(); pi) {
    FOR_EACH_EL_RF_IF(el, rf, pi->_slew[el][rf]) {
      pin._relax_slew(nullptr, el, rf, el, rf, *(pi->_slew[el][rf]));
    }
  }
  
  // Relax the slew from its fanin.
  for(auto arc : pin._fanin) {
    arc->_fprop_slew();
  }
}

// Procedure: _fprop_delay
void Timer::_fprop_delay(Pin& pin) {

  // clear delay
  for(auto arc : pin._fanin) {
    arc->_reset_delay();
  }

  // Compute the delay from its fanin.
  for(auto arc : pin._fanin) {
    arc->_fprop_delay();
  }
}

// Procedure: _fprop_at
void Timer::_fprop_at(Pin& pin) {
  
  // clear at
  pin._reset_at();

  // PI
  if(auto pi = pin.primary_input(); pi) {
    FOR_EACH_EL_RF_IF(el, rf, pi->_at[el][rf]) {
      pin._relax_at(nullptr, el, rf, el, rf, *(pi->_at[el][rf]));
    }
  }

  // Relax the at from its fanin.
  for(auto arc : pin._fanin) {
    arc->_fprop_at();
  }
}

// Procedure: _fprop_test
void Timer::_fprop_test(Pin& pin) {
  
  // reset tests
  for(auto test : pin._tests) {
    test->_reset();
  }
  
  // Obtain the rat
  if(!_clocks.empty()) {

    // Update the rat
    for(auto test : pin._tests) {
      // TODO: currently we assume a single clock...
      test->_fprop_rat(_clocks.begin()->second._period);
      
      // compute the cppr credit if any
      if(_cppr_analysis) {
        FOR_EACH_EL_RF_IF(el, rf, test->raw_slack(el, rf)) {
          test->_cppr_credit[el][rf] = _cppr_credit(*test, el, rf);
        }
      }
    }
  }
}

// Procedure: _bprop_rat
void Timer::_bprop_rat(Pin& pin) {

  pin._reset_rat();

  // PO
  if(auto po = pin.primary_output(); po) {
    FOR_EACH_EL_RF_IF(el, rf, po->_rat[el][rf]) {
      pin._relax_rat(nullptr, el, rf, el, rf, *(po->_rat[el][rf]));
    }
  }

  // Test
  for(auto test : pin._tests) {
    FOR_EACH_EL_RF_IF(el, rf, test->_rat[el][rf]) {
      if(test->_cppr_credit[el][rf]) {
        pin._relax_rat(
          &test->_arc, el, rf, el, rf, *test->_rat[el][rf] + *test->_cppr_credit[el][rf]
        );
      }
      else {
        pin._relax_rat(&test->_arc, el, rf, el, rf, *test->_rat[el][rf]);
      }
    }
  }

  // Relax the rat from its fanout.
  for(auto arc : pin._fanout) {
    arc->_bprop_rat();
  }
}

// Procedure: _build_fprop_cands
// Performs DFS to find all nodes in the fanout cone of frontiers.
void Timer::_build_fprop_cands(Pin& from) {
  
  assert(!from._has_state(Pin::FPROP_CAND) && !from._has_state(Pin::IN_FPROP_STACK));

  from._insert_state(Pin::FPROP_CAND | Pin::IN_FPROP_STACK);

  for(auto arc : from._fanout) {
    if(auto& to = arc->_to; !to._has_state(Pin::FPROP_CAND)) {
      _build_fprop_cands(to);
    }
    else if(to._has_state(Pin::IN_FPROP_STACK)) {
      _scc_analysis = true;
    }
  }
  
  _fprop_cands.push_front(&from);  // insert from front for scc traversal
  from._remove_state(Pin::IN_FPROP_STACK);
}

// Procedure: _build_bprop_cands
// Perform the DFS to find all nodes in the fanin cone of fprop candidates.
void Timer::_build_bprop_cands(Pin& to) {
  
  assert(!to._has_state(Pin::BPROP_CAND) && !to._has_state(Pin::IN_BPROP_STACK));

  to._insert_state(Pin::BPROP_CAND | Pin::IN_BPROP_STACK);

  // add pin to scc
  if(_scc_analysis && to._has_state(Pin::FPROP_CAND) && !to._scc) {
    _scc_cands.push_back(&to);
  }

  for(auto arc : to._fanin) {
    if(auto& from=arc->_from; !from._has_state(Pin::BPROP_CAND)) {
      _build_bprop_cands(from);
    }
  }
  
  _bprop_cands.push_front(&to);
  to._remove_state(Pin::IN_BPROP_STACK);
}

// Procedure: _build_prop_cands
void Timer::_build_prop_cands() {

  _scc_analysis = false;

  // Discover all fprop candidates.
  for(const auto& ftr : _frontiers) {
    if(ftr->_has_state(Pin::FPROP_CAND)) {
      continue;
    }
    _build_fprop_cands(*ftr);
  }

  // Discover all bprop candidates.
  for(auto fcand : _fprop_cands) {

    if(fcand->_has_state(Pin::BPROP_CAND)) {
      continue;
    }

    _scc_cands.clear();
    _build_bprop_cands(*fcand);

    if(!_scc_analysis) {
      assert(_scc_cands.empty());
    }
    
    // here dfs returns with exacly one scc if exists
    if(auto& c = _scc_cands; c.size() >= 2 || (c.size() == 1 && c[0]->has_self_loop())) {
      auto& scc = _insert_scc(c);
      scc._unloop();
    }
  }
}

// Procedure: _build_prop_tasks
void Timer::_build_prop_tasks() {
  
  // explore propagation candidates
  _build_prop_cands();

  // Emplace the fprop task
  // (1) propagate the rc timing
  // (2) propagate the slew 
  // (3) propagate the delay
  // (4) propagate the arrival time.
  for(auto pin : _fprop_cands) {
    assert(!pin->_ftask);
    pin->_ftask = _taskflow.emplace([this, pin] () {
      _fprop_rc_timing(*pin);
      _fprop_slew(*pin);
      _fprop_delay(*pin);
      _fprop_at(*pin);
      _fprop_test(*pin);
    }).name(pin->_name);
  }
  
  // Build the dependency
  for(auto to : _fprop_cands) {
    for(auto arc : to->_fanin) {
      if(arc->_has_state(Arc::LOOP_BREAKER)) {
        continue;
      }
      if(auto& from = arc->_from; from._has_state(Pin::FPROP_CAND)) {
        from._ftask->precede(to->_ftask.value());
      }
    }
  }

  // Emplace the bprop task
  // (1) propagate the required arrival time
  for(auto pin : _bprop_cands) {
    assert(!pin->_btask);
    pin->_btask = _taskflow.emplace([this, pin] () {
      _bprop_rat(*pin);
    }).name(pin->_name);
  }

  // Build the task dependencies.
  for(auto to : _bprop_cands) {
    for(auto arc : to->_fanin) {
      if(arc->_has_state(Arc::LOOP_BREAKER)) {
        continue;
      }
      if(auto& from = arc->_from; from._has_state(Pin::BPROP_CAND)) {
        to->_btask->precede(from._btask.value());
      }
    } 
  }

  // Connect with ftasks
  for(auto pin : _bprop_cands) {
    if(pin->_btask->num_dependents() == 0 && pin->_ftask) {
      pin->_ftask->precede(pin->_btask.value()); 
    }
  }

}

// Procedure: _clear_prop_tasks
void Timer::_clear_prop_tasks() {
  
  // fprop is a subset of bprop
  for(auto pin : _bprop_cands) {
    pin->_ftask.reset();
    pin->_btask.reset();
    pin->_remove_state();
  }

  _fprop_cands.clear();
  _bprop_cands.clear();
}

// Function: update_timing
// Perform comprehensive timing update: 
// (1) grpah-based timing (GBA)
// (2) path-based timing (PBA)
void Timer::update_timing() {
  std::scoped_lock lock(_mutex);
  _update_timing();
}

// Function: _update_timing
void Timer::_update_timing() {
  
  // Timing is update-to-date
  if(!_lineage) {
    assert(_frontiers.size() == 0);
    return;
  }

  // materialize the lineage
  _executor.run(_taskflow).wait();
  _taskflow.clear();
  _lineage.reset();
  
  // always enbale full timing update 
  _insert_full_timing_frontiers();

  // build propagation tasks
  _build_prop_cands();

  // debug the graph
  // _taskflow.dump(std::cout);

  // rebuild ftask part of _taskflow
  _rebuild_ftask(); 

  // Execute the ftask
  _executor.run(_taskflow).wait();

  // initialize vivekDAG only for btask
  _initialize_vivekDAG();

  // export a csr format of vivekDAG 
  _export_csr();

  // call cuda kernel
  // call_cuda_topo_centric_vector();
  call_cuda_partition();
  
  // run btask sequentially 
  _run_topo_gpu();

  /*
  // partition vivekDAG
  auto start = std::chrono::steady_clock::now();
  _partition_vivekDAG_GDCA();
  auto end = std::chrono::steady_clock::now();
  _partition_DAG_time += std::chrono::duration_cast<std::chrono::microseconds>(end-start).count();

  // rebuild _taskflow by vivekDAG
  start = std::chrono::steady_clock::now();
  _rebuild_taskflow_GDCA();
  end = std::chrono::steady_clock::now();
  _vivek_btask_rebuild_time += std::chrono::duration_cast<std::chrono::microseconds>(end-start).count();
  // Execute the task
  start = std::chrono::steady_clock::now();
  _executor.run(_taskflow).wait();
  end = std::chrono::steady_clock::now();
  _vivek_btask_runtime += std::chrono::duration_cast<std::chrono::microseconds>(end-start).count();
  */

  _taskflow.clear();

  // _print_max_parallelism();

  // Clear vivek partition parameters
  _reset_partition();

  // Clear the propagation tasks.
  _clear_prop_tasks();

  // Clear frontiers
  _clear_frontiers();

  // clear the state
  _remove_state();
}

// Procedure: _update_area
void Timer::_update_area() {
  
  _update_timing();

  if(_has_state(AREA_UPDATED)) {
    return;
  }
  
  _area = 0.0f;

  for(const auto& kvp : _gates) {
    if(const auto& c = kvp.second._cell[MIN]; c->area) {
      _area = *_area + *c->area;
    }
    else {
      OT_LOGE("cell ", c->name, " has no area defined");
      _area.reset();
      break;
    }
  }

  _insert_state(AREA_UPDATED);
}

// Procedure: _update_power
void Timer::_update_power() {

  _update_timing();

  if(_has_state(POWER_UPDATED)) {
    return;
  }

  // Update the static leakage power
  _leakage_power = 0.0f;
  
  for(const auto& kvp : _gates) {
    if(const auto& c = kvp.second._cell[MIN]; c->leakage_power) {
      _leakage_power = *_leakage_power + *c->leakage_power;
    }
    else {
      OT_LOGE("cell ", c->name, " has no leakage_power defined");
      _leakage_power.reset();
      break;
    }
  }

  _insert_state(POWER_UPDATED);
}

// Procedure: _update_endpoints
void Timer::_update_endpoints() {

  _update_timing();

  if(_has_state(EPTS_UPDATED)) {
    return;
  }

  // reset the storage and build task
  FOR_EACH_EL_RF(el, rf) {

    _endpoints[el][rf].clear();
    
    _taskflow.emplace([this, el=el, rf=rf] () {

      // for each po
      for(auto& po : _pos) {
        if(po.second.slack(el, rf).has_value()) {
          _endpoints[el][rf].emplace_back(el, rf, po.second);
        }
      }

      // for each test
      for(auto& test : _tests) {
        if(test.slack(el, rf).has_value()) {
          _endpoints[el][rf].emplace_back(el, rf, test);
        }
      }
      
      // sort endpoints
      std::sort(_endpoints[el][rf].begin(), _endpoints[el][rf].end());

      // update the worst negative slack (wns)
      if(!_endpoints[el][rf].empty()) {
        _wns[el][rf] = _endpoints[el][rf].front().slack();
      }
      else {
        _wns[el][rf] = std::nullopt;
      }

      // update the tns, and fep
      if(!_endpoints[el][rf].empty()) {
        _tns[el][rf] = 0.0f;
        _fep[el][rf] = 0;
        for(const auto& ept : _endpoints[el][rf]) {
          if(auto slack = ept.slack(); slack < 0.0f) {
            _tns[el][rf] = *_tns[el][rf] + slack;
            _fep[el][rf] = *_fep[el][rf] + 1; 
          }
        }
      }
      else {
        _tns[el][rf] = std::nullopt;
        _fep[el][rf] = std::nullopt;
      }
    });
  }

  // run tasks
  _executor.run(_taskflow).wait();
  _taskflow.clear();

  _insert_state(EPTS_UPDATED);
}

// Function: tns
// Update the total negative slack for any transition and timing split. The procedure applies
// the parallel reduction to compute the value.
std::optional<float> Timer::report_tns(std::optional<Split> el, std::optional<Tran> rf) {

  std::scoped_lock lock(_mutex);

  _update_endpoints();

  std::optional<float> v;

  if(!el && !rf) {
    FOR_EACH_EL_RF_IF(s, t, _tns[s][t]) {
      v = !v ? _tns[s][t] : *v + *(_tns[s][t]);
    }
  }
  else if(el && !rf) {
    FOR_EACH_RF_IF(t, _tns[*el][t]) {
      v = !v ? _tns[*el][t] : *v + *(_tns[*el][t]);
    }
  }
  else if(!el && rf) {
    FOR_EACH_EL_IF(s, _tns[s][*rf]) {
      v = !v ? _tns[s][*rf] : *v + *(_tns[s][*rf]);
    }
  }
  else {
    v = _tns[*el][*rf];
  }

  return v;
}

// Function: wns
// Update the total negative slack for any transition and timing split. The procedure apply
// the parallel reduction to compute the value.
std::optional<float> Timer::report_wns(std::optional<Split> el, std::optional<Tran> rf) {

  std::scoped_lock lock(_mutex);

  _update_endpoints();

  std::optional<float> v;
  
  if(!el && !rf) {
    FOR_EACH_EL_RF_IF(s, t, _wns[s][t]) {
      v = !v ? _wns[s][t] : std::min(*v, *(_wns[s][t]));
    }
  }
  else if(el && !rf) {
    FOR_EACH_RF_IF(t, _wns[*el][t]) {
      v = !v ? _wns[*el][t] : std::min(*v, *(_wns[*el][t]));
    }
  }
  else if(!el && rf) {
    FOR_EACH_EL_IF(s, _wns[s][*rf]) {
      v = !v ? _wns[s][*rf] : std::min(*v, *(_wns[s][*rf]));
    }
  }
  else {
    v = _wns[*el][*rf];
  }

  return v;
}

// Function: fep
// Update the failing end points
std::optional<size_t> Timer::report_fep(std::optional<Split> el, std::optional<Tran> rf) {
  
  std::scoped_lock lock(_mutex);

  _update_endpoints();

  std::optional<size_t> v;

  if(!el && !rf) {
    FOR_EACH_EL_RF_IF(s, t, _fep[s][t]) {
      v = !v ? _fep[s][t] : *v + *(_fep[s][t]);
    }
  }
  else if(el && !rf) {
    FOR_EACH_RF_IF(t, _fep[*el][t]) {
      v = !v ? _fep[*el][t] : *v + *(_fep[*el][t]);
    }
  }
  else if(!el && rf) {
    FOR_EACH_EL_IF(s, _fep[s][*rf]) {
      v = !v ? _fep[s][*rf] : *v + *(_fep[s][*rf]);
    }
  }
  else {
    v = _fep[*el][*rf];
  }

  return v;
}

// Function: leakage_power
std::optional<float> Timer::report_leakage_power() {
  std::scoped_lock lock(_mutex);
  _update_power();
  return _leakage_power;
}

// Function: area
// Sum up the area of each gate in the design.
std::optional<float> Timer::report_area() {
  std::scoped_lock lock(_mutex);
  _update_area();
  return _area;
}
    
// Procedure: _enable_full_timing_update
void Timer::_enable_full_timing_update() {
  _insert_state(FULL_TIMING);
}

// Procedure: _insert_full_timing_frontiers
void Timer::_insert_full_timing_frontiers() {

  // insert all zero-fanin pins to the frontier list
  for(auto& kvp : _pins) {
    _insert_frontier(kvp.second);
  }

  // clear the rc-net update flag
  for(auto& kvp : _nets) {
    kvp.second._rc_timing_updated = false;
  }
}

// Procedure: _insert_frontier
void Timer::_insert_frontier(Pin& pin) {
  
  if(pin._frontier_satellite) {
    return;
  }

  pin._frontier_satellite = _frontiers.insert(_frontiers.end(), &pin);
  
  // reset the scc.
  if(pin._scc) {
    _remove_scc(*pin._scc);
  }
}

// Procedure: _remove_frontier
void Timer::_remove_frontier(Pin& pin) {
  if(pin._frontier_satellite) {
    _frontiers.erase(*pin._frontier_satellite);
    pin._frontier_satellite.reset();
  }
}

// Procedure: _clear_frontiers
void Timer::_clear_frontiers() {
  for(auto& ftr : _frontiers) {
    ftr->_frontier_satellite.reset();
  }
  _frontiers.clear();
}

// Procedure: _insert_scc
SCC& Timer::_insert_scc(std::vector<Pin*>& cands) {
  
  // create scc only of size at least two
  auto& scc = _sccs.emplace_front(std::move(cands));
  scc._satellite = _sccs.begin();

  return scc;
}

// Procedure: _remove_scc
void Timer::_remove_scc(SCC& scc) {
  assert(scc._satellite);
  scc._clear();
  _sccs.erase(*scc._satellite); 
}

// Function: report_at   
// Report the arrival time in picoseconds at a given pin name.
std::optional<float> Timer::report_at(const std::string& name, Split m, Tran t) {
  std::scoped_lock lock(_mutex);
  return _report_at(name, m, t);
}

// Function: _report_at
std::optional<float> Timer::_report_at(const std::string& name, Split m, Tran t) {
  _update_timing();
  if(auto itr = _pins.find(name); itr != _pins.end() && itr->second._at[m][t]) {
    return itr->second._at[m][t]->numeric;
  }
  else return std::nullopt;
}

// Function: report_rat
// Report the required arrival time in picoseconds at a given pin name.
std::optional<float> Timer::report_rat(const std::string& name, Split m, Tran t) {
  std::scoped_lock lock(_mutex);
  return _report_rat(name, m, t);
}

// Function: _report_rat
std::optional<float> Timer::_report_rat(const std::string& name, Split m, Tran t) {
  _update_timing();
  if(auto itr = _pins.find(name); itr != _pins.end() && itr->second._at[m][t]) {
    return itr->second._rat[m][t];
  }
  else return std::nullopt;
}

// Function: report_slew
// Report the slew in picoseconds at a given pin name.
std::optional<float> Timer::report_slew(const std::string& name, Split m, Tran t) {
  std::scoped_lock lock(_mutex);
  return _report_slew(name, m, t);
}

// Function: _report_slew
std::optional<float> Timer::_report_slew(const std::string& name, Split m, Tran t) {
  _update_timing();
  if(auto itr = _pins.find(name); itr != _pins.end() && itr->second._slew[m][t]) {
    return itr->second._slew[m][t]->numeric;
  }
  else return std::nullopt;
}

// Function: report_slack
std::optional<float> Timer::report_slack(const std::string& pin, Split m, Tran t) {
  std::scoped_lock lock(_mutex);
  return _report_slack(pin, m, t);
}

// Function: _report_slack
std::optional<float> Timer::_report_slack(const std::string& pin, Split m, Tran t) {
  _update_timing();
  if(auto itr = _pins.find(pin); itr != _pins.end()) {
    return itr->second.slack(m, t);
  }
  else return std::nullopt;
}

// Function: report_load
// Report the load at a given pin name
std::optional<float> Timer::report_load(const std::string& name, Split m, Tran t) {
  std::scoped_lock lock(_mutex);
  return _report_load(name, m, t);
}

// Function: _report_load
std::optional<float> Timer::_report_load(const std::string& name, Split m, Tran t) {
  _update_timing();
  if(auto itr = _nets.find(name); itr != _nets.end()) {
    return itr->second._load(m, t);
  }
  else return std::nullopt;
}

// Function: set_at
Timer& Timer::set_at(std::string name, Split m, Tran t, std::optional<float> v) {

  std::scoped_lock lock(_mutex);

  auto task = _taskflow.emplace([this, name=std::move(name), m, t, v] () {
    if(auto itr = _pis.find(name); itr != _pis.end()) {
      _set_at(itr->second, m, t, v);
    }
    else {
      OT_LOGE("can't set at (PI ", name, " not found)");
    }
  });

  _add_to_lineage(task);

  return *this;
}

// Procedure: _set_at
void Timer::_set_at(PrimaryInput& pi, Split m, Tran t, std::optional<float> v) {
  pi._at[m][t] = v;
  _insert_frontier(pi._pin);
}

// Function: set_rat
Timer& Timer::set_rat(std::string name, Split m, Tran t, std::optional<float> v) {

  std::scoped_lock lock(_mutex);
  
  auto op = _taskflow.emplace([this, name=std::move(name), m, t, v] () {
    if(auto itr = _pos.find(name); itr != _pos.end()) {
      _set_rat(itr->second, m, t, v);
    }
    else {
      OT_LOGE("can't set rat (PO ", name, " not found)");
    }
  });

  _add_to_lineage(op);

  return *this;
}

// Procedure: _set_rat
void Timer::_set_rat(PrimaryOutput& po, Split m, Tran t, std::optional<float> v) {
  po._rat[m][t] = v;
  _insert_frontier(po._pin);
}

// Function: set_slew
Timer& Timer::set_slew(std::string name, Split m, Tran t, std::optional<float> v) {

  std::scoped_lock lock(_mutex);
  
  auto task = _taskflow.emplace([this, name=std::move(name), m, t, v] () {
    if(auto itr = _pis.find(name); itr != _pis.end()) {
      _set_slew(itr->second, m, t, v);
    }
    else {
      OT_LOGE("can't set slew (PI ", name, " not found)");
    }
  });

  _add_to_lineage(task);

  return *this;
}

// Procedure: _set_slew
void Timer::_set_slew(PrimaryInput& pi, Split m, Tran t, std::optional<float> v) {
  pi._slew[m][t] = v;
  _insert_frontier(pi._pin);
}

// Function: set_load
Timer& Timer::set_load(std::string name, Split m, Tran t, std::optional<float> v) {

  std::scoped_lock lock(_mutex);
  
  auto task = _taskflow.emplace([this, name=std::move(name), m, t, v] () {
    if(auto itr = _pos.find(name); itr != _pos.end()) {
      _set_load(itr->second, m, t, v);
    }
    else {
      OT_LOGE("can't set load (PO ", name, " not found)");
    }
  });

  _add_to_lineage(task);

  return *this;
}

// Procedure: _set_load
void Timer::_set_load(PrimaryOutput& po, Split m, Tran t, std::optional<float> v) {

  po._load[m][t] = v ? *v : 0.0f;

  // Update the net load
  if(auto net = po._pin._net) {
    net->_rc_timing_updated = false;
  }
  
  // Enable the timing propagation.
  for(auto arc : po._pin._fanin) {
    _insert_frontier(arc->_from);
  }
  _insert_frontier(po._pin);
}

void Timer::_initialize_local_crit_cost_pins() {

  /*
   * To calculate the critical path of a node in a task dependency DAG,
   * you can use Depth-First Search (DFS) with topological sorting to
   * find the longest path leading to that node.
   */

  // _f(b)prop_cands stores topological order of pins
  std::deque<Pin*> fprop_cands_copy = _fprop_cands;
  std::deque<Pin*> bprop_cands_copy = _bprop_cands;

  // calculate prev_crit_cost + self_cost for ftask
  for(auto pin : fprop_cands_copy) {
    if(pin->_fanin.size() == 0) {
      pin->_ftemp_cost_prev_self = pin->_fself_cost; // for pin with 0 fanin in ftask, prev_crit_cost = 0;
    }
  }

  while(!fprop_cands_copy.empty()) {
    Pin* p = fprop_cands_copy.front();
    fprop_cands_copy.pop_front();
    for(auto arc : p->_fanout) {
      Pin* to = &(arc->_to);
      int new_cost = p->_ftemp_cost_prev_self + to->_fself_cost;
      if(new_cost > to->_ftemp_cost_prev_self) {
        to->_ftemp_cost_prev_self = new_cost;
      }
    }
  }

  fprop_cands_copy = _fprop_cands; // refill topo-queue

  // calculate after_crit_cost + self_cost for btask
  for(auto pin : fprop_cands_copy) {
    if(pin->_fanin.size() == 0) {
      pin->_btemp_cost_after_self = pin->_bself_cost; // for pin with 0 fanin in btask, after_crit_cost = 0;
    }
  }

  while(!fprop_cands_copy.empty()) {
    Pin* p = fprop_cands_copy.front();
    fprop_cands_copy.pop_front();
    for(auto arc : p->_fanout) {
      Pin* to = &(arc->_to);
      int new_cost = p->_btemp_cost_after_self + to->_bself_cost;
      if(new_cost > to->_btemp_cost_after_self) {
        to->_btemp_cost_after_self = new_cost;
      }
    }
  }

  // calculate after_crit_cost + self cost for ftask
  for(auto pin : bprop_cands_copy) {
    if(pin->_fanout.size() == 0) {
      pin->_ftemp_cost_after_self = pin->_btemp_cost_after_self + pin->_fself_cost; // for pin with 0 fanout in ftask,
                                                                                    // after_crit_cost = its after cost in btask portion
    }
  }

  while(!bprop_cands_copy.empty()) {
    Pin* p = bprop_cands_copy.front();
    bprop_cands_copy.pop_front();
    for(auto arc : p->_fanin) {
      Pin* from = &(arc->_from);
      int new_cost = p->_ftemp_cost_after_self + from->_fself_cost;
      if(new_cost > from->_ftemp_cost_after_self) {
        from->_ftemp_cost_after_self = new_cost;
      }
    }
  }

  bprop_cands_copy = _bprop_cands; // refill topo-queue

  // calculate prev_crit_cost + self cost for btask
  for(auto pin : bprop_cands_copy) {
    if(pin->_fanout.size() == 0) {
      pin->_btemp_cost_prev_self = pin->_ftemp_cost_prev_self + pin->_bself_cost; // for pin with 0 fanout in btask,
                                                                                  // prev_crit_cost = its prev cost in ftask portion
    }
  }

  while(!bprop_cands_copy.empty()) {
    Pin* p = bprop_cands_copy.front();
    bprop_cands_copy.pop_front();
    for(auto arc : p->_fanin) {
      Pin* from = &(arc->_from);
      int new_cost = p->_btemp_cost_prev_self + from->_bself_cost;
      if(new_cost > from->_btemp_cost_prev_self) {
        from->_btemp_cost_prev_self = new_cost;
      }
    }
  }

  for(auto pin : _frontiers) {
    pin->_flocal_crit_cost = pin->_ftemp_cost_prev_self + pin->_ftemp_cost_after_self - pin->_fself_cost;
    pin->_blocal_crit_cost = pin->_btemp_cost_prev_self + pin->_btemp_cost_after_self - pin->_bself_cost;
    pin->_fdepth = pin->_ftemp_cost_prev_self;
    pin->_bdepth = pin->_btemp_cost_prev_self;
  }

}

bool Timer::_form_cycle(VivekTask* vtask1, VivekTask* vtask2) {

  /*
   * use dfs to check if there is more than 1 paths between vtask1 and vtask2, if so, merge 
   * vtask1 and vtask2 will form cycle
   */
  
  // mark all vtask as unvisited for dfs checking cycle
  for(size_t i=0; i<_vivekDAG._vtask_ptrs.size(); i++) {
    _vivekDAG._vtask_ptrs[i]->_num_visited = 0;
  }
  
  // find number of paths from vtask1 to vtask2, i.e., number of times vtask2 being visited
  _dfs_form_cycle(vtask1, vtask2);
  if(vtask2->_num_visited > 1) {
    return true;
  }

  return false;
}

void Timer::_initialize_vivekDAG() {

  // initialize vivekDAG at its most fine_grained
  
  // initialize pin's vid 
  int id = 0;
//  for(auto pin : _fprop_cands) {
//    pin->_fvid = id;
//    id++;
//  }
//
  for(auto pin : _bprop_cands) {
    pin->_bvid = id;
    id++;
  }

  // initialize local critical path costs for each pin 
  _initialize_local_crit_cost_pins();

  // add tasks to vivekDAG
  for(auto pin : _bprop_cands) {
    _vivekDAG.addVivekTask(pin->_bvid, pin->_bdepth, std::make_pair(false, pin));  
  }

  // update self cost of tasks in vivekDAG for later cost update
  for(auto task : _vivekDAG._vtask_ptrs) {
    task->getSelfCost();
  }

  // add dependencies of vtasks in vivekDAG
  for(auto pin : _bprop_cands) { // NOTICE: for vbtask, fanin and fanout of dependencies and circuits are reverse
    int cur_vtask = pin->_bvid;
    for(auto arc : pin->_fanout) {
      Pin* to = &(arc->_to);
      int to_id = to->_bvid;
      _vivekDAG._vtask_ptrs[cur_vtask]->addFanin(to_id); 
    }
    for(auto arc : pin->_fanin) {
      Pin* from = &(arc->_from);
      int from_id = from->_bvid;
      _vivekDAG._vtask_ptrs[cur_vtask]->addFanout(from_id); 
    }
  }
  
  // remove replicate fanin/fanout of tasks in vivekDAG
  for(size_t i=0; i<_vivekDAG._vtask_ptrs.size(); i++) {
    _vivekDAG._vtask_ptrs[i]->deleteRepFan();
  }

  // add tasks to global priority queue
  for(size_t i=0; i<_vivekDAG._vtask_ptrs.size(); i++) {
    _global_task_vector.push_back(_vivekDAG._vtask_ptrs[i]);
  }
  std::sort(_global_task_vector.begin(), _global_task_vector.end(), [](const VivekTask* a, VivekTask* b){ 
    return a->_local_crit_cost < b->_local_crit_cost;
  });

  // _vtask_ptrs stores the topological order of all the vtasks
  for(auto pin : _bprop_cands) {
    _top_down_topo_order_cur_vivekDAG_vector.push_back(pin->_bvid); 
  }

  /*
  // check if initialization is correct
  std::cerr << "print vivekDAG to check if initialization is correct.\n";
  for(size_t i=0; i<_vivekDAG._vtask_ptrs.size(); i++) {
    if(i <= _fprop_cands.size() - 1) {
      std::cerr << "pin name(" << i << "): " << _fprop_cands[i]->_name << "(" << _vivekDAG._vtask_ptrs[i]->_local_crit_cost << ")\n";
    }
    else {
      std::cerr << "pin name(" << i << "): " << _bprop_cands[i - _fprop_cands.size()]->_name << "(" << _vivekDAG._vtask_ptrs[i]->_local_crit_cost << ")\n";
    }
    std::cerr << "pin->_fanin: ";
    for(auto id : _vivekDAG._vtask_ptrs[i]->_fanin) {
      std::cerr << id << " ";
    }
    std::cerr << "\n";
    std::cerr << "pin->_fanout: ";
    for(auto id : _vivekDAG._vtask_ptrs[i]->_fanout) {
      std::cerr << id << " ";
    }
    std::cerr << "\n";
  }
  */
}

void Timer::_export_csr() {

  // check DAG
  // // clear original taskflow
  // _taskflow.clear();

  // // emplace all tasks in vivekDAG to _taskflow
  // for(auto task : _vivekDAG._vtask_ptrs) {
  //   task->_tftask = _taskflow.emplace([this, task] () {
  //     auto start = std::chrono::steady_clock::now();
  //     for(auto pair : task->_pins) {
  //       if(pair.first) {
  //         _fprop_rc_timing(*(pair.second));
  //         _fprop_slew(*(pair.second));
  //         _fprop_delay(*(pair.second));
  //         _fprop_at(*(pair.second));
  //         _fprop_test(*(pair.second));
  //       }
  //       else {
  //         _bprop_rat(*(pair.second));
  //       }
  //     }
  //     auto end = std::chrono::steady_clock::now();
  //     task->_runtime = std::chrono::duration_cast<std::chrono::microseconds>(end-start).count();
  //   // }).name(std::to_string(task->_id));
  //   }).name(task->_pins[0].second->_name + "(" + std::to_string(task->_id) + ")");
  // }

  // // add dependencies in _taskflow
  // for(size_t task_id=0; task_id<_vivekDAG._vtask_ptrs.size(); task_id++) {
  //   for(auto successor_id : _vivekDAG._vtask_ptrs[task_id]->_fanout) {
  //       _vivekDAG._vtask_ptrs[task_id]->_tftask.precede(_vivekDAG._vtask_ptrs[successor_id]->_tftask);
  //   }
  // } 

  // _taskflow.dump(std::cout);
  // _taskflow.clear();

  /*
   * export csr matrices of vivekDAG
   */
  _adjp.push_back(0);
  size_t index = 0;
  for(size_t i=1; i<_vivekDAG._vtask_ptrs.size(); i++) {
    index = index + _vivekDAG._vtask_ptrs[i-1]->_fanout.size();
    if(_vivekDAG._vtask_ptrs[i]->_fanout.size() == 0) { // for those who has no fanout, set its fanout to -1
      _adjp.push_back(-1);  
    }
    else {
      _adjp.push_back(index); 
    }
  }   
  for(auto vtask : _vivekDAG._vtask_ptrs) {
    _adjncy.insert(_adjncy.end(), vtask->_fanout.begin(), vtask->_fanout.end());
  }
  for(size_t i=0; i<_vivekDAG._vtask_ptrs.size(); i++) {
    _adjncy_size.push_back(_vivekDAG._vtask_ptrs[i]->_fanout.size()); 
  }
  for(size_t i=0; i<_vivekDAG._vtask_ptrs.size(); i++) {
    _dep_size.push_back(_vivekDAG._vtask_ptrs[i]->_fanin.size()); 
  }

  // check csr
  // std::cerr << "_adjp = [";
  // for(auto id : _adjp) {
  //   std::cerr << id << " ";
  // }
  // std::cerr << "]\n";
  // std::cerr << "_adjncy = [";
  // for(auto id : _adjncy) {
  //   std::cerr << id << " ";
  // }
  // std::cerr << "]\n";
  // std::cerr << "_adjncy_size = [";
  // for(auto id : _adjncy_size) {
  //   std::cerr << id << " ";
  // }
  // std::cerr << "]\n";
  // std::cerr << "_dep_size = [";
  // for(auto id : _dep_size) {
  //   std::cerr << id << " ";
  // }
  // std::cerr << "]\n";
}

void Timer::_run_topo_gpu() {

  std::cout << "_topo_result_gpu.size() = " << _topo_result_gpu.size() << "\n";
  for(auto task_id : _topo_result_gpu) {
    std::pair<bool, Pin*> p = _vivekDAG._vtask_ptrs[task_id]->_pins[0];  
    // std::cout << p.second->_name << "(" << task_id << ")\n";
    _bprop_rat(*(p.second));
  } 

  /*
  for(int task_id=0; task_id<_vivekDAG._vtask_ptrs.size(); task_id++) {
    std::pair<bool, Pin*> p = _vivekDAG._vtask_ptrs[task_id]->_pins[0];  
    std::cout << p.second->_name << "(" << task_id << ")\n";
    _bprop_rat(*(p.second));
  } 
  */

}

void Timer::_partition_vivekDAG() {

  // merging parameter
  size_t num_merge = 0;
  size_t cur_task = 0;
  size_t merge_cnt = 0;
  while(1) {

    if(cur_task > _vivekDAG._vtasks.size() - 1 || merge_cnt >= num_merge) {
      break;
    }

    /*
     * step 2-1. choose starting task for merging
     */
    auto start = std::chrono::steady_clock::now();
    // choose least_cost_task from _global_task_vector to merge
    VivekTask* least_cost_task = _global_task_vector[cur_task];
    cur_task = cur_task + 1;
    
    // check if this task has been merged
    if(least_cost_task->_merged) {
      continue;
    }
    auto end = std::chrono::steady_clock::now();
    _choose_least_cost_task_time += std::chrono::duration_cast<std::chrono::microseconds>(end-start).count();

    /*
     * step 2-2. find merging candidates
     */
    start = std::chrono::steady_clock::now();
    // get dependent and successor tasks of this task and store into local queue
    std::vector<VivekTask*> local_task_vector;

    for(auto dependent_id : least_cost_task->_fanin) {
      if(!_vivekDAG._vtask_ptrs[dependent_id]->_merged) { // skip merged neighbor
        // std::cerr << "push " << dependent_id << "(" << _vivekDAG._vtask_ptrs[dependent_id]->_self_cost << ") to local queue\n";
        local_task_vector.push_back(_vivekDAG._vtask_ptrs[dependent_id]);
      }
    }
    for(auto successor_id : least_cost_task->_fanout) {
      if(!_vivekDAG._vtask_ptrs[successor_id]->_merged) {
        // std::cerr << "push " << successor_id << "(" << _vivekDAG._vtask_ptrs[successor_id]->_self_cost << ") to local queue\n";
        local_task_vector.push_back(_vivekDAG._vtask_ptrs[successor_id]);
      }
    }

    // sort the neighbors according to self_cost
    std::sort(local_task_vector.begin(), local_task_vector.end(), [](const VivekTask* a, VivekTask* b){
      return a->_local_crit_cost < b->_local_crit_cost;
    });
    end = std::chrono::steady_clock::now();
    _find_merge_candidates_time += std::chrono::duration_cast<std::chrono::microseconds>(end-start).count();

    /*
     * step 2-3. try merging
     */
    start = std::chrono::steady_clock::now();
    _try_merging(least_cost_task, local_task_vector, merge_cnt);
    end = std::chrono::steady_clock::now();
    _try_merging_time += std::chrono::duration_cast<std::chrono::microseconds>(end-start).count();
  }
}

void Timer::_GDCA_dfs() {

  for(auto vtask : _vivekDAG._vtask_ptrs) {
    if(vtask->_fanin.size() == 0) {
      _global_task_queue_GDCA.push(vtask); 
    }
  } 

  // merging parameter
  size_t dst_cluster_size = 1; // destination cluster size
  size_t cur_cluster_id = 0; // current cluster id  
  // std::list<int> boundary; // vtasks(id) whose dependents are not fully released
  while(!_global_task_queue_GDCA.empty()) {

    // choose least_cost_task from _global_task_vector to merge
    VivekTask* master = _global_task_queue_GDCA.top();
    _global_task_queue_GDCA.pop();
    
    // a vector to store the vtasks in the same cluster
    std::vector<VivekTask*> cur_cluster;

    // assign cluster id
    master->_cluster_id = cur_cluster_id;
    cur_cluster.push_back(master); 

    // release dependents for successors of master
    for(auto successor_id : master->_fanout) {
      if(!_vivekDAG._vtask_ptrs[successor_id]->_merged) {
        _vivekDAG._vtask_ptrs[successor_id]->_num_deps_release++;
        // if dependents of the successors are all released, put it into ready to be merged
        if(_vivekDAG._vtask_ptrs[successor_id]->_num_deps_release == _vivekDAG._vtask_ptrs[successor_id]->_fanin.size()) {
          _global_task_queue_GDCA.push(_vivekDAG._vtask_ptrs[successor_id]);
        }
      } 
    }
   
    size_t cluster_size = 1;
    
    while(cluster_size < dst_cluster_size && !_global_task_queue_GDCA.empty()) {
     
      VivekTask* next = _global_task_queue_GDCA.top();
      _global_task_queue_GDCA.pop();
      next->_cluster_id = cur_cluster_id; 
      cur_cluster.push_back(next);
      cluster_size++;

      for(auto successor_id : next->_fanout) {
        _vivekDAG._vtask_ptrs[successor_id]->_num_deps_release++;
        if(_vivekDAG._vtask_ptrs[successor_id]->_num_deps_release == _vivekDAG._vtask_ptrs[successor_id]->_fanin.size()) {
          _global_task_queue_GDCA.push(_vivekDAG._vtask_ptrs[successor_id]);
        }
      }
    }
    _vivekDAG._vtask_clusters.push_back(cur_cluster);
    cur_cluster_id++;

  } 
}

void Timer::_GDCA_build_coarsen_graph() {

  // each cluster in _vtask_clusters stands for a new vtask
  for(size_t cluster_id=0; cluster_id<_vivekDAG._vtask_clusters.size(); cluster_id++) {
    int rebuild_vtask_id = cluster_id;
    int rebuild_vtask_local_cost = 0;
    std::vector<std::pair<bool, Pin*>> rebuild_vtask_pins; 
    for(auto vtask : _vivekDAG._vtask_clusters[cluster_id]) {
      rebuild_vtask_pins.insert(rebuild_vtask_pins.end(), vtask->_pins.begin(), vtask->_pins.end());
      // rebuild_vtask_pins.push_back(vtask->_pins[0]); // in GDCA, each vtask only has one pin
    }
    _rebuild_vivekDAG.addVivekTask(rebuild_vtask_id, rebuild_vtask_local_cost, rebuild_vtask_pins);
  }

 
  /*
  // add fanin according to all fanin of vtask inside a cluster
  for(size_t cluster_id=0; cluster_id<_vivekDAG._vtask_clusters.size(); cluster_id++) {
    for(auto vtask : _vivekDAG._vtask_clusters[cluster_id]) {
      for(auto fanin_id : vtask->_fanin) {
        if(_vivekDAG._vtask_ptrs[fanin_id]->_cluster_id != cluster_id) {
          _rebuild_vivekDAG._vtask_ptrs[cluster_id]->addFanin(_vivekDAG._vtask_ptrs[fanin_id]->_cluster_id);
        }
      }
    }
  }
  */

  // add fanout according to all fanout of vtask inside a cluster
  for(size_t cluster_id=0; cluster_id<_vivekDAG._vtask_clusters.size(); cluster_id++) {
    for(auto vtask : _vivekDAG._vtask_clusters[cluster_id]) {
      for(auto fanout_id : vtask->_fanout) {
        if(_vivekDAG._vtask_ptrs[fanout_id]->_cluster_id != cluster_id) {
          _rebuild_vivekDAG._vtask_ptrs[cluster_id]->addFanout(_vivekDAG._vtask_ptrs[fanout_id]->_cluster_id);
        }
      }
    }
  }
 
  // delete replicate fanin/fanout
  for(size_t i=0; i<_rebuild_vivekDAG._vtask_ptrs.size(); i++) {
    _rebuild_vivekDAG._vtask_ptrs[i]->deleteRepFan();
  } 

  std::cout << "before partition, graph size: " << _vivekDAG._vtask_ptrs.size() << "\n";
  std::cout << "after GDCA partition, graph size: " << _rebuild_vivekDAG._vtask_ptrs.size() << "\n";
}

void Timer::_GDCA_build_coarsen_graph_par() {

  // clear _taskflow for building coarsen graph
  _taskflow.clear();
 
  // each cluster is a tftask 
  std::vector<tf::Task> tftasks(_vivekDAG._vtask_clusters.size());
  std::vector<std::pair<bool, Pin*>> empty_pins;
  for(size_t cluster_id=0; cluster_id<_vivekDAG._vtask_clusters.size(); cluster_id++) {
    _rebuild_vivekDAG.addVivekTask(cluster_id, 0, empty_pins);
  }
  for(size_t cluster_id=0; cluster_id<_vivekDAG._vtask_clusters.size(); cluster_id++) {
    tftasks[cluster_id] = _taskflow.emplace([this, cluster_id](){
      for(auto vtask : _vivekDAG._vtask_clusters[cluster_id]) {
        std::vector<std::pair<bool, Pin*>> rebuild_vtask_pins; 
        _rebuild_vivekDAG._vtask_ptrs[cluster_id]->addPin(vtask->_pins[0]);
        for(auto fanout_id : vtask->_fanout) {
          if(_vivekDAG._vtask_ptrs[fanout_id]->_cluster_id != cluster_id) {
            _rebuild_vivekDAG._vtask_ptrs[cluster_id]->addFanout(_vivekDAG._vtask_ptrs[fanout_id]->_cluster_id);
          }
        }
      }
    });  
  } 

  // add dependencies through _taskflow
  _executor.run(_taskflow).wait();

}

void Timer::_partition_vivekDAG_GDCA() {

  /*
   * improve vivek partition by removing checking cycle 
   */

  auto start = std::chrono::steady_clock::now();
  _GDCA_dfs();
  auto end = std::chrono::steady_clock::now();
  GDCA_dfs_time += std::chrono::duration_cast<std::chrono::microseconds>(end-start).count();
  
  start = std::chrono::steady_clock::now();
  _GDCA_build_coarsen_graph();
  end = std::chrono::steady_clock::now();
  GDCA_build_coarsen_graph_time += std::chrono::duration_cast<std::chrono::microseconds>(end-start).count();

}

void Timer::_try_merging(VivekTask* least_cost_task, std::vector<VivekTask*>& local_task_vector, size_t& merge_cnt) {

  /*
   * step 3. try merging
   */

  // get neighbor with least local cost task from local queue
  size_t cur_neighbor = 0;
  while(1) {

    if(local_task_vector.size() == 0 || cur_neighbor > local_task_vector.size() - 1) {
      break; 
    }
   
    /*
     * step 2-3-1. choose least_cost_neighbor from local_task_vector to merge
     */ 
    auto start = std::chrono::steady_clock::now();
    VivekTask* least_cost_neighbor = local_task_vector[cur_neighbor];
    cur_neighbor = cur_neighbor + 1;
    
    auto end = std::chrono::steady_clock::now();
    _choose_least_cost_neighbor_time += std::chrono::duration_cast<std::chrono::microseconds>(end-start).count();

    
    /*
     * step 2-3-2. perform merge
     */
    start = std::chrono::steady_clock::now();
    // check who is dependent, when checking cycle, only need to check from node to its fanout(cuz _taskflow has not cycle originally)
    // check if merging with this neighbor leads to cycle, i.e., is there more than 1 route between its neighbor and itself
    bool neighbor_is_fanout = false;
    if(std::find(least_cost_task->_fanout.begin(), least_cost_task->_fanout.end(), least_cost_neighbor->_id) != least_cost_task->_fanout.end()) {
      neighbor_is_fanout = true;
      if(_form_cycle(least_cost_task, least_cost_neighbor)) {
        // std::cerr << "cycle detected.\n";
        continue;
      }
    }
    else {
      neighbor_is_fanout = false;
      if(_form_cycle(least_cost_neighbor, least_cost_task)) {
        // std::cerr << "cycle detected.\n";
        continue;
      }
    }

    _merge_vivekTasks(least_cost_task, least_cost_neighbor, neighbor_is_fanout);
    merge_cnt = merge_cnt + 1;
    end = std::chrono::steady_clock::now();
    _perform_merge_time += std::chrono::duration_cast<std::chrono::microseconds>(end-start).count();

    /*
     * step 2-3-3. update local critical path cost 
     */
    start = std::chrono::steady_clock::now();
    // after merged, update _top_down_topo_order_cur_vivekDAG_vector for _update_local_crit_cost() & rebuild_taskflow()
    if(neighbor_is_fanout) { // if least_cost_neighbor is fanout, replace least_cost_neighbor with merged_id(_vtask_ptrs.size()-1)
      std::replace(_top_down_topo_order_cur_vivekDAG_vector.begin(),
                   _top_down_topo_order_cur_vivekDAG_vector.end(),
                   least_cost_neighbor->_id, _vivekDAG._vtask_ptrs[_vivekDAG._vtask_ptrs.size()-1]->_id
                  );
    }
    else{
    // if least_cost_neighbor is fanin, replace least_cost_task with merged_id(_vtask_ptrs.size()-1)
      std::replace(_top_down_topo_order_cur_vivekDAG_vector.begin(),
                   _top_down_topo_order_cur_vivekDAG_vector.end(),
                   least_cost_task->_id, _vivekDAG._vtask_ptrs[_vivekDAG._vtask_ptrs.size()-1]->_id
                  );
    }
    // update_local_crit_cost
    _update_local_crit_cost();
    end = std::chrono::steady_clock::now();
    _update_top_down_vector += std::chrono::duration_cast<std::chrono::microseconds>(end-start).count();

    /*
     * step 2-3-4. update _global_task_vector 
     */
    start = std::chrono::steady_clock::now();
    _insert_merged_vivekTask(_vivekDAG._vtask_ptrs[_vivekDAG._vtask_ptrs.size()-1]);
    end = std::chrono::steady_clock::now();
    _update_global_task_vector += std::chrono::duration_cast<std::chrono::microseconds>(end-start).count();

    break;
  }
}

void Timer::_partition_timing_profile() {

  std::cout << "***************** partition timing profile ************************\n";
  std::cout << "_partition runtime(step 2): " << _partition_DAG_time << "\n";
  std::cout << "_choose_least_cost_task_time(step 2-1): " << _choose_least_cost_task_time << "\n";
  std::cout << "_find_merge_candidates_time(step 2-2): " << _find_merge_candidates_time << "\n";
  std::cout << "_try_merging_time(step 2-3): " << _try_merging_time << "\n";
  std::cout << "_choose_least_cost_neighbor_time(step 2-3-1): " << _choose_least_cost_neighbor_time << "\n";
  std::cout << "_perform_merge_time(step 2-3-2): " << _perform_merge_time << "\n";
  std::cout << "_update_top_down_vector(step 2-3-3): " << _update_top_down_vector << "\n";
  std::cout << "_update_global_task_vector(step 2-3-4): " << _update_global_task_vector << "\n";
  std::cout << "********************************************************************\n";

}

void Timer::_print_max_parallelism() {

  /*
   * max_parallelisim = total runtime of DAG / runtime of critical path of DAG
   */

  size_t DAG_runtime = 0;
  for(auto task : _vivekDAG._vtask_ptrs) {
    DAG_runtime += task->_runtime;
  }
  
  // the topological order is stored in _rebuild_vivekDAG._vtask_ptrs 
  for(auto vtask : _rebuild_vivekDAG._vtask_ptrs) { // calculate self_cost
    vtask->_self_cost = vtask->_runtime;
  }
  for(auto vtask : _rebuild_vivekDAG._vtask_ptrs) { // initialize prev_crit_cost
    vtask->_prev_crit_cost = vtask->_self_cost;
  }
  for(auto vtask : _rebuild_vivekDAG._vtask_ptrs) {
    for(auto successor_id : vtask->_fanout) {
      int new_cost = vtask->_prev_crit_cost + _rebuild_vivekDAG._vtask_ptrs[successor_id]->_self_cost;
      if(new_cost > _rebuild_vivekDAG._vtask_ptrs[successor_id]->_prev_crit_cost) {
        _rebuild_vivekDAG._vtask_ptrs[successor_id]->_prev_crit_cost = new_cost;
      }
    }
  }
  int crit_path_runtime = 0;
  for(auto vtask : _rebuild_vivekDAG._vtask_ptrs) {
    if(crit_path_runtime < vtask->_prev_crit_cost) {
      crit_path_runtime = vtask->_prev_crit_cost;
    }
  }
  std::cout << "max_parallelism = " << static_cast<double> (DAG_runtime) / crit_path_runtime << "\n"; 
}

void Timer::_dfs_form_cycle(VivekTask* start, VivekTask* end) {
  
  start->_num_visited ++;

  if(end->_num_visited > 1) {
    return;
  }

  // check if reach end
  if(start->_id == end->_id) {
    return;
  }

  for(auto successor_id : start->_fanout) {
    if(!_vivekDAG._vtask_ptrs[successor_id]->_merged) {
      if(successor_id == end->_id || _vivekDAG._vtask_ptrs[successor_id]->_num_visited == 0) {
        _dfs_form_cycle(_vivekDAG._vtask_ptrs[successor_id], end);
      }
    }
  }

}

void Timer::_merge_vivekTasks(VivekTask* vtask1, VivekTask* vtask2, bool neighbor_is_fanout) {
  
  if(!neighbor_is_fanout) { // if vtask2 is fanin, then switch it. 
                            // always make sure vtask1 is fanin for the rest operations
    VivekTask* temp = vtask1;
    vtask1 = vtask2;
    vtask2 = temp;
  } 

//  std::cerr << "2 tasks to be merged: " << vtask1->_id << " and " << vtask2->_id << " to " << _vivekDAG._vtask_ptrs.size() << "\n"; 

  // id for merged task
  int merged_id = _vivekDAG._vtask_ptrs.size(); 

  // local critical cost for merged task will be calculated after it is added into the vivekDAG 
  int merged_local_crit_cost = 0;

  // pins for merged task
  std::vector<std::pair<bool, Pin*>> merged_pins;
  merged_pins = vtask1->_pins;
  merged_pins.insert(merged_pins.end(), vtask2->_pins.begin(), vtask2->_pins.end());  
//  std::cerr << "is fanout\n";

  // add merged task to vivekDAG
  _vivekDAG.addVivekTask(merged_id, merged_local_crit_cost, merged_pins);

  // handle dependencies for merged task
  // add fanin and fanout of vtask1 and vtask2 to merged task
  for(auto dependent_id : vtask1->_fanin) {
    if(!_vivekDAG._vtask_ptrs[dependent_id]->_merged) {
      _vivekDAG._vtask_ptrs[merged_id]->addFanin(dependent_id);
    }
  }
  for(auto dependent_id : vtask2->_fanin) {
    if(!_vivekDAG._vtask_ptrs[dependent_id]->_merged && dependent_id != vtask1->_id) {
      _vivekDAG._vtask_ptrs[merged_id]->addFanin(dependent_id);
    }
  }
  for(auto successor_id : vtask1->_fanout) {
    if(!_vivekDAG._vtask_ptrs[successor_id]->_merged && successor_id != vtask2->_id) {
      _vivekDAG._vtask_ptrs[merged_id]->addFanout(successor_id);
    }
  }
  for(auto successor_id : vtask2->_fanout) {
    if(!_vivekDAG._vtask_ptrs[successor_id]->_merged) {
      _vivekDAG._vtask_ptrs[merged_id]->addFanout(successor_id);
    }
  }
  _vivekDAG._vtask_ptrs[merged_id]->deleteRepFan(); // remove redundency in fanin/fanout

  // add dependencies for merged task's fanin&fanout
  for(auto dependent_id : _vivekDAG._vtask_ptrs[merged_id]->_fanin) {
    if(!_vivekDAG._vtask_ptrs[dependent_id]->_merged) {
      _vivekDAG._vtask_ptrs[dependent_id]->addFanout(merged_id);
    }
  }
  for(auto successor_id : _vivekDAG._vtask_ptrs[merged_id]->_fanout) {
    if(!_vivekDAG._vtask_ptrs[successor_id]->_merged) {
      _vivekDAG._vtask_ptrs[successor_id]->addFanin(merged_id);
    }
  }

  // indicate vtask1 and vtask2 as merged after the operations, i.e., they are removed from vivekDAG
  vtask1->_merged = true;
  vtask2->_merged = true;
  
  // update self cost of merged task
  _vivekDAG._vtask_ptrs[merged_id]->getSelfCost();
//  std::cerr << "task " << merged_id << ": self cost = " << _vivekDAG._vtask_ptrs[merged_id]->_self_cost << "\n";
//   std::cerr << "fanin: ";
//  for(auto index : _vivekDAG._vtask_ptrs[merged_id]->_fanin) {
//    std::cerr << index << " ";
//  }
//  std::cerr << "\n";
//  std::cerr << "fanout: ";
//  for(auto index : _vivekDAG._vtask_ptrs[merged_id]->_fanout) {
//    std::cerr << index << " ";
//  }
//  std::cerr << "\n";
//  std::cerr << "merged pins: ";
//  for(auto& pair : _vivekDAG._vtask_ptrs[merged_id]->_pins) {
//    std::cerr << pair.second->_name << " ";
//  }
//  std::cerr << "\n";

}

void Timer::_insert_merged_vivekTask(VivekTask* merged_vtask) {

  /*
   * traverse _global_task_vector and insert merged_vtask into the correct position 
   * by _local_crit_cost(increasing order)
   */
 
  auto it = std::lower_bound(_global_task_vector.begin(), _global_task_vector.end(), merged_vtask,  [](const VivekTask* a, const VivekTask* b) {
    return a->_local_crit_cost < b->_local_crit_cost;
  });
	_global_task_vector.insert(it, merged_vtask);	
  
}

void Timer::_update_local_crit_cost() {

  // get topological order of current vivekDAG
  // mark all vtask as unvisited for dfs get topological order 
  // for(size_t i=0; i<_vivekDAG._vtask_ptrs.size(); i++) {
  //   _vivekDAG._vtask_ptrs[i]->_num_visited = 0;
  // }

  std::stack<int> top_down_topo_order; // topological order from top to bottom
  std::queue<int> bottom_up_topo_order; // topological order from bottom to top
  /*
  for(size_t i=0; i<_vivekDAG._vtask_ptrs.size(); i++) {
    if(!_vivekDAG._vtask_ptrs[i]->_merged) {
      if(_vivekDAG._vtask_ptrs[i]->_fanin.size() == 0 && _vivekDAG._vtask_ptrs[i]->_num_visited == 0) {
        _dfs_topo_order(i, top_down_topo_order, bottom_up_topo_order);
      }
    }
  }
  */
  for(int id = _top_down_topo_order_cur_vivekDAG_vector.size()-1; id >=0; id --) {
    if(!_vivekDAG._vtask_ptrs[_top_down_topo_order_cur_vivekDAG_vector[id]]->_merged) {
      top_down_topo_order.push(_top_down_topo_order_cur_vivekDAG_vector[id]);
      bottom_up_topo_order.push(_top_down_topo_order_cur_vivekDAG_vector[id]);
    }
  }

  // store top to bottom topological order of current vivekDAG for later use 
  // _top_down_topo_order_cur_vivekDAG = top_down_topo_order;

  // from top down topological order, calculate vtask's prev_crit_cost(including self_cost)
  for(auto task : _vivekDAG._vtask_ptrs) {
    if(!task->_merged) {
      if(task->_fanin.size() == 0) { // for task with 0 fanin, its prev_crit_cost = 0;
        task->_prev_crit_cost = task->_self_cost;
      }
    }
  }

  while(!top_down_topo_order.empty()) {
    int cur_id = top_down_topo_order.top();
    top_down_topo_order.pop();
    for(auto successor_id : _vivekDAG._vtask_ptrs[cur_id]->_fanout) {
      if(!_vivekDAG._vtask_ptrs[successor_id]->_merged) {
        int new_cost = _vivekDAG._vtask_ptrs[successor_id]->_self_cost + _vivekDAG._vtask_ptrs[cur_id]->_prev_crit_cost; 
        if(new_cost > _vivekDAG._vtask_ptrs[successor_id]->_prev_crit_cost) {
           _vivekDAG._vtask_ptrs[successor_id]->_prev_crit_cost = new_cost;
        }
      }
    }
  }
  /*
  for(auto cur_id : _top_down_topo_order_cur_vivekDAG_vector) {
    for(auto successor_id : _vivekDAG._vtask_ptrs[cur_id]->_fanout) {
      if(!_vivekDAG._vtask_ptrs[successor_id]->_merged) {
        int new_cost = _vivekDAG._vtask_ptrs[successor_id]->_self_cost + _vivekDAG._vtask_ptrs[cur_id]->_prev_crit_cost; 
        if(new_cost > _vivekDAG._vtask_ptrs[successor_id]->_prev_crit_cost) {
           _vivekDAG._vtask_ptrs[successor_id]->_prev_crit_cost = new_cost;
        }
      }
    }
  }
  */

  // from bottom up topological order, calculate vtask's after_crit_cost(including self_cost)
  for(auto task : _vivekDAG._vtask_ptrs) {
    if(!task->_merged) {
      if(task->_fanout.size() == 0) { // for task with 0 fanout, its after_crit_cost = 0;
        task->_after_crit_cost = task->_self_cost;
      }
    }
  }

  while(!bottom_up_topo_order.empty()) {
    int cur_id = bottom_up_topo_order.front();
    bottom_up_topo_order.pop();
    for(auto dependent_id : _vivekDAG._vtask_ptrs[cur_id]->_fanin) {
      if(!_vivekDAG._vtask_ptrs[dependent_id]->_merged) {
        int new_cost = _vivekDAG._vtask_ptrs[dependent_id]->_self_cost + _vivekDAG._vtask_ptrs[cur_id]->_after_crit_cost; 
        if(new_cost > _vivekDAG._vtask_ptrs[dependent_id]->_after_crit_cost) {
           _vivekDAG._vtask_ptrs[dependent_id]->_after_crit_cost = new_cost;
        }
      }
    }
  }
  /*
  for(int cur_id = _top_down_topo_order_cur_vivekDAG_vector.size()-1; cur_id >= 0; cur_id--) {
    for(auto dependent_id : _vivekDAG._vtask_ptrs[cur_id]->_fanin) {
      if(!_vivekDAG._vtask_ptrs[dependent_id]->_merged) {
        int new_cost = _vivekDAG._vtask_ptrs[dependent_id]->_self_cost + _vivekDAG._vtask_ptrs[cur_id]->_after_crit_cost;
        if(new_cost > _vivekDAG._vtask_ptrs[dependent_id]->_after_crit_cost) {
           _vivekDAG._vtask_ptrs[dependent_id]->_after_crit_cost = new_cost;
        }
      }
    }
  }
  */

  // update local critical path cost for all tasks in vivekDAG
  for(auto task : _vivekDAG._vtask_ptrs) {
    if(!task->_merged) {
      task->_local_crit_cost = task->_prev_crit_cost + task->_after_crit_cost - task->_self_cost;
    }
  }
}

void Timer::_dfs_topo_order(int start, std::stack<int>& top_down_result, std::queue<int>& bottom_up_result) {

  _vivekDAG._vtask_ptrs[start]->_num_visited++; 

  for(auto successor_id : _vivekDAG._vtask_ptrs[start]->_fanout) {
    if(!_vivekDAG._vtask_ptrs[successor_id]->_merged) {
      if(_vivekDAG._vtask_ptrs[successor_id]->_num_visited == 0) {
        _dfs_topo_order(successor_id, top_down_result, bottom_up_result); 
      }
    }
  }

  top_down_result.push(start);
  bottom_up_result.push(start);

}

void Timer::_rebuild_ftask() {

  _taskflow.clear();

  // Emplace the fprop task
  // (1) propagate the rc timing
  // (2) propagate the slew 
  // (3) propagate the delay
  // (4) propagate the arrival time.
  for(auto pin : _fprop_cands) {
    assert(!pin->_ftask);
    pin->_ftask = _taskflow.emplace([this, pin] () {
      _fprop_rc_timing(*pin);
      _fprop_slew(*pin);
      _fprop_delay(*pin);
      _fprop_at(*pin);
      _fprop_test(*pin);
    });
  }
  
  // Build the dependency
  for(auto to : _fprop_cands) {
    for(auto arc : to->_fanin) {
      if(arc->_has_state(Arc::LOOP_BREAKER)) {
        continue;
      }
      if(auto& from = arc->_from; from._has_state(Pin::FPROP_CAND)) {
        from._ftask->precede(to->_ftask.value());
      }
    }
  }  

}

void Timer::_rebuild_taskflow_vivek() {

  // clear original taskflow
  _taskflow.clear();

  // emplace all tasks in vivekDAG to _taskflow
  for(auto task : _vivekDAG._vtask_ptrs) {
    if(!task->_merged) {
      task->_tftask = _taskflow.emplace([this, task] () {
        auto start = std::chrono::steady_clock::now();
        for(auto pair : task->_pins) {
          if(pair.first) {
            _fprop_rc_timing(*(pair.second));
            _fprop_slew(*(pair.second));
            _fprop_delay(*(pair.second));
            _fprop_at(*(pair.second));
            _fprop_test(*(pair.second));
          }
          else {
            _bprop_rat(*(pair.second));
          }
        }
        auto end = std::chrono::steady_clock::now();
        task->_runtime = std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count();
      }).name(std::to_string(task->_cluster_id));  
    }
  }

  for(auto task_id : _top_down_topo_order_cur_vivekDAG_vector) {
    if(!_vivekDAG._vtask_ptrs[task_id]->_merged) {
      for(auto successor_id : _vivekDAG._vtask_ptrs[task_id]->_fanout) {
        if(!_vivekDAG._vtask_ptrs[successor_id]->_merged) {
          _vivekDAG._vtask_ptrs[task_id]->_tftask.precede(_vivekDAG._vtask_ptrs[successor_id]->_tftask);
        }
      }
    }
  }
}

void Timer::_rebuild_taskflow_GDCA() {
  
  // clear original taskflow
  _taskflow.clear();

  // emplace all tasks in vivekDAG to _taskflow
  for(auto task : _rebuild_vivekDAG._vtask_ptrs) {
    task->_tftask = _taskflow.emplace([this, task] () {
      auto start = std::chrono::steady_clock::now();
      for(auto pair : task->_pins) {
        if(pair.first) {
          _fprop_rc_timing(*(pair.second));
          _fprop_slew(*(pair.second));
          _fprop_delay(*(pair.second));
          _fprop_at(*(pair.second));
          _fprop_test(*(pair.second));
        }
        else {
          _bprop_rat(*(pair.second));
        }
      }
      auto end = std::chrono::steady_clock::now();
      task->_runtime = std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count();
    }).name(std::to_string(task->_id));
  }
  
  // add dependencies in _taskflow
  for(size_t task_id=0; task_id<_rebuild_vivekDAG._vtask_ptrs.size(); task_id++) {
    for(auto successor_id : _rebuild_vivekDAG._vtask_ptrs[task_id]->_fanout) {
        _rebuild_vivekDAG._vtask_ptrs[task_id]->_tftask.precede(_rebuild_vivekDAG._vtask_ptrs[successor_id]->_tftask);
    }
  }
  
}

void Timer::_run_vivekDAG_GDCA_seq() {

  // run vivekDAG sequentially
  for(auto cluster : _vivekDAG._vtask_clusters) {
    for(auto vtask : cluster) {
      for(auto pair : vtask->_pins) {
        if(pair.first) {
          _fprop_rc_timing(*(pair.second));
          _fprop_slew(*(pair.second));
          _fprop_delay(*(pair.second));
          _fprop_at(*(pair.second));
          _fprop_test(*(pair.second));
        }
        else {
          _bprop_rat(*(pair.second));
        }
      }
    }
  }

}

void Timer::_task_timing_profile() {

  // std::cout << "task_runtime = [";
  // for(auto task : _vivekDAG._vtask_ptrs) {
  //   if(!task->_merged) {
  //     std::cout << task->_runtime << ", ";  
  //   }
  // }
  // std::cout << "]\n"; 
 
  size_t task_cnt = 0;
  for(auto task : _vivekDAG._vtask_ptrs) {
    if(!task->_merged) {
      task_cnt = task_cnt + 1;
    }
  }
  std::cout << "before, num of tasks = " << _bprop_cands.size() << "\n";
  std::cout << "after merging, num of tasks = " << task_cnt << "\n";

}

void Timer::_reset_partition() {

  _vivekDAG.resetVivekDAG();
  _rebuild_vivekDAG.resetVivekDAG();

//  std::cerr << "--------------------------- reseting vivekDAG ----------------------------\n";

//  std::cerr << _vivekDAG._vtask_ptrs.size() << "\n";

  _top_down_topo_order_cur_vivekDAG = std::stack<int>(); // topological order from top to bottom for current DAG
  _top_down_topo_order_cur_vivekDAG_vector.clear();

  // clear priority queue
  _global_task_queue = std::priority_queue<VivekTask*, std::vector<VivekTask*>, CompareTaskByCost>();
  _global_task_queue_GDCA = std::priority_queue<VivekTask*, std::vector<VivekTask*>, CompareTaskByCost>();
  _global_task_vector.clear();
  _global_task_vector_GDCA.clear();

  _adjp.clear(); 
  _adjncy.clear();
  _adjncy_size.clear();
  _dep_size.clear(); // number of dependents of each node
  
  _topo_result_cpu.clear();
  _topo_result_gpu.clear();
  _partition_result_gpu.clear();

  for(auto pin : _frontiers) {
    pin->_fvisited = false;
    pin->_bvisited = false;
    pin->_ftemp_cost_prev_self = 0;
    pin->_ftemp_cost_after_self = 0;
    pin->_btemp_cost_after_self = 0;
    pin->_btemp_cost_prev_self = 0;
    pin->_flocal_crit_cost = 0;
    pin->_blocal_crit_cost = 0;
    pin->_fvid = 0;
    pin->_bvid = 0;
  }
}

};  // end of namespace ot. -----------------------------------------------------------------------




