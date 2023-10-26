#ifndef OT_TIMER_TIMER_HPP_
#define OT_TIMER_TIMER_HPP_

#include <ot/timer/gate.hpp>
#include <ot/timer/pin.hpp>
#include <ot/timer/arc.hpp>
#include <ot/timer/net.hpp>
#include <ot/timer/test.hpp>
#include <ot/timer/clock.hpp>
#include <ot/timer/endpoint.hpp>
#include <ot/timer/path.hpp>
#include <ot/timer/sfxt.hpp>
#include <ot/timer/pfxt.hpp>
#include <ot/timer/cppr.hpp>
#include <ot/timer/scc.hpp>
#include <ot/static/logger.hpp>
#include <ot/spef/spef.hpp>
#include <ot/verilog/verilog.hpp>
#include <ot/sdc/sdc.hpp>
#include <ot/tau/tau15.hpp>

#include <ot/timer/vivek.hpp>

namespace ot {

// Class: Timer
class Timer {

  struct CompareTaskByCost {
    bool operator()(const VivekTask* task1, const VivekTask* task2) const {
      return task1->_local_crit_cost < task2->_local_crit_cost;
    }
  };

  friend class Shell;
  
  constexpr static int FULL_TIMING   = 0x01;
  constexpr static int EPTS_UPDATED  = 0x02;
  constexpr static int AREA_UPDATED  = 0x04;
  constexpr static int POWER_UPDATED = 0x08;

  public:
   
    ~Timer() {

      std::cout << "CPU_topo(partition)_runtime: " << CPU_topo_runtime << "\n";
      std::cout << "GPU_topo(partition)_runtime: " << GPU_topo_runtime << "\n";
      std::cout << "_vivek_btask_rebuild_time: " << _vivek_btask_rebuild_time << "\n";
      std::cout << "_vivek_btask_runtime: " << _vivek_btask_runtime << "\n";

    }

    // Builder
    Timer& set_num_threads(unsigned);
    Timer& read_celllib(std::filesystem::path, std::optional<Split> = {});
    Timer& read_verilog(std::filesystem::path);
    Timer& read_spef(std::filesystem::path);
    Timer& read_sdc(std::filesystem::path);
    Timer& read_timing(std::filesystem::path);
    Timer& insert_net(std::string);
    Timer& insert_gate(std::string, std::string);
    Timer& repower_gate(std::string, std::string);
    Timer& remove_net(std::string);
    Timer& remove_gate(std::string);
    Timer& disconnect_pin(std::string);
    Timer& connect_pin(std::string, std::string);
    Timer& insert_primary_input(std::string);
    Timer& insert_primary_output(std::string);
    Timer& set_at(std::string, Split, Tran, std::optional<float>);
    Timer& set_rat(std::string, Split, Tran, std::optional<float>);
    Timer& set_slew(std::string, Split, Tran, std::optional<float>);
    Timer& set_load(std::string, Split, Tran, std::optional<float>);
    Timer& create_clock(std::string, float);
    Timer& create_clock(std::string, std::string, float);
    Timer& cppr(bool);
    Timer& set_time_unit(second_t);
    Timer& set_capacitance_unit(farad_t);
    Timer& set_resistance_unit(ohm_t);
    Timer& set_voltage_unit(volt_t);
    Timer& set_power_unit(watt_t);
    Timer& set_current_unit(ampere_t);

    // Action.
    void update_timing();

    std::optional<float> report_at(const std::string&, Split, Tran);
    std::optional<float> report_rat(const std::string&, Split, Tran);
    std::optional<float> report_slew(const std::string&, Split, Tran);
    std::optional<float> report_slack(const std::string&, Split, Tran);
    std::optional<float> report_load(const std::string&, Split, Tran);
    std::optional<float> report_area();
    std::optional<float> report_leakage_power();
    std::optional<float> report_tns(std::optional<Split> = {}, std::optional<Tran> = {});
    std::optional<float> report_wns(std::optional<Split> = {}, std::optional<Tran> = {});
    std::optional<size_t> report_fep(std::optional<Split> = {}, std::optional<Tran> = {});
    
    std::vector<Path> report_timing(size_t);
    std::vector<Path> report_timing(size_t, Split);
    std::vector<Path> report_timing(size_t, Tran);
    std::vector<Path> report_timing(size_t, Split, Tran);
    std::vector<Path> report_timing(PathGuide);

    // Accessor
    void dump_graph(std::ostream&) const;
    void dump_power(std::ostream&) const;
    void dump_taskflow(std::ostream&) const;
    void dump_cell(std::ostream&, const std::string&, Split) const;
    void dump_celllib(std::ostream&, Split) const;
    void dump_net_load(std::ostream&) const;
    void dump_pin_cap(std::ostream&) const;
    void dump_at(std::ostream&) const;
    void dump_rat(std::ostream&) const;
    void dump_slew(std::ostream&) const;
    void dump_slack(std::ostream&) const;
    void dump_timer(std::ostream&) const;
    void dump_verilog(std::ostream&, const std::string&) const;
    void dump_spef(std::ostream&) const;
    void dump_rctree(std::ostream&) const;
    
    inline auto num_primary_inputs() const;
    inline auto num_primary_outputs() const;
    inline auto num_pins() const;
    inline auto num_nets() const;
    inline auto num_arcs() const;
    inline auto num_gates() const;
    inline auto num_tests() const;
    inline auto num_sccs() const;
    inline auto time_unit() const;
    inline auto power_unit() const;
    inline auto resistance_unit() const;
    inline auto current_unit() const;
    inline auto voltage_unit() const;
    inline auto capacitance_unit() const;
    inline std::optional<float> cell_voltage() const;
    
    inline const auto& primary_inputs() const;
    inline const auto& primary_outputs() const;
    inline const auto& pins() const;
    inline const auto& nets() const;
    inline const auto& gates() const;
    inline const auto& clocks() const;
    inline const auto& tests() const;
    inline const auto& arcs() const;

    /*
     * GDCA partition
     */
    void call_cuda_topo_2queue();
    void call_cuda_topo_centric_vector();
    void call_cuda_partition();
    void bfs_cpu(std::vector<int>& distance, std::vector<int>& parent);
    void topo_cpu(std::vector<int>& dep_size);
    int partition_size = 1;
    void partition_cpu(std::vector<int>& dep_size, std::vector<int>& partition_result_cpu, std::vector<int>& partition_counter_cpu, int* max_partition_id); 

  private:

    mutable std::shared_mutex _mutex;

    tf::Taskflow _taskflow;
    tf::Executor _executor;

    int _state {0};
    
    bool _scc_analysis {false};

    std::optional<tf::Task> _lineage;
    std::optional<CpprAnalysis> _cppr_analysis;
    std::optional<second_t> _time_unit;
    std::optional<watt_t> _power_unit;
    std::optional<ohm_t> _resistance_unit;
    std::optional<farad_t> _capacitance_unit;
    std::optional<ampere_t> _current_unit;
    std::optional<volt_t> _voltage_unit;

    TimingData<std::optional<Celllib>, MAX_SPLIT> _celllib;

    std::unordered_map<std::string, PrimaryInput> _pis;
    std::unordered_map<std::string, PrimaryOutput> _pos; 
    std::unordered_map<std::string, Pin> _pins;
    std::unordered_map<std::string, Net> _nets;
    std::unordered_map<std::string, Gate> _gates;
    std::unordered_map<std::string, Clock> _clocks;
 
    std::list<Test> _tests;
    std::list<Arc> _arcs;
    std::list<Pin*> _frontiers;
    std::list<SCC> _sccs;

    TimingData<std::vector<Endpoint>, MAX_SPLIT, MAX_TRAN> _endpoints;
    TimingData<std::optional<float>,  MAX_SPLIT, MAX_TRAN> _wns;
    TimingData<std::optional<float>,  MAX_SPLIT, MAX_TRAN> _tns;
    TimingData<std::optional<size_t>, MAX_SPLIT, MAX_TRAN> _fep;
    
    std::optional<float> _area;
    std::optional<float> _leakage_power;

    std::deque<Pin*> _fprop_cands;
    std::deque<Pin*> _bprop_cands;

    IndexGenerator<size_t> _pin_idx_gen {0u};
    IndexGenerator<size_t> _arc_idx_gen {0u};
    
    std::vector<Pin*> _scc_cands;
    std::vector<Pin*> _idx2pin;
    std::vector<Arc*> _idx2arc;

    std::vector<Endpoint*> _worst_endpoints(size_t);
    std::vector<Endpoint*> _worst_endpoints(size_t, Split);
    std::vector<Endpoint*> _worst_endpoints(size_t, Tran);
    std::vector<Endpoint*> _worst_endpoints(size_t, Split, Tran);
    std::vector<Endpoint*> _worst_endpoints(const PathGuide&);

    std::vector<Path> _report_timing(std::vector<Endpoint*>&&, size_t);
    
    bool _is_redundant_timing(const Timing&, Split) const;

    void _to_time_unit(const second_t&);
    void _to_capacitance_unit(const farad_t&);
    void _to_resistance_unit(const ohm_t&);
    void _to_power_unit(const watt_t&);
    void _to_current_unit(const ampere_t&);
    void _to_voltage_unit(const volt_t&);
    void _add_to_lineage(tf::Task);
    void _rebase_unit(Celllib&);
    void _rebase_unit(spef::Spef&);
    void _update_timing();
    void _update_endpoints();
    void _update_area();
    void _update_power();
    void _fprop_rc_timing(Pin&);
    void _fprop_slew(Pin&);
    void _fprop_delay(Pin&);
    void _fprop_at(Pin&);
    void _fprop_test(Pin&);
    void _bprop_rat(Pin&);
    void _build_prop_cands();
    void _build_fprop_cands(Pin&);
    void _build_bprop_cands(Pin&);
    void _build_prop_tasks();
    void _clear_prop_tasks();
    void _read_spef(spef::Spef&);;
    void _verilog(vlog::Module&);
    void _timing(tau15::Timing&);
    void _read_sdc(sdc::SDC&);
    void _read_sdc(sdc::SetInputDelay&);
    void _read_sdc(sdc::SetInputTransition&);
    void _read_sdc(sdc::SetOutputDelay&);
    void _read_sdc(sdc::SetLoad&);
    void _read_sdc(sdc::CreateClock&);
    void _connect_pin(Pin&, Net&);
    void _disconnect_pin(Pin&);
    void _insert_frontier(Pin&);
    void _remove_frontier(Pin&);
    void _remove_scc(SCC&);
    void _clear_frontiers();
    void _insert_primary_output(const std::string&);
    void _insert_primary_input(const std::string&);
    void _insert_gate(const std::string&, const std::string&);
    void _insert_gate_arcs(Gate&);
    void _remove_gate_arcs(Gate&);
    void _repower_gate(const std::string&, const std::string&);
    void _remove_gate(Gate&);
    void _remove_net(Net&);
    void _remove_pin(Pin&);
    void _remove_arc(Arc&);
    void _remove_test(Test&);
    void _set_at(PrimaryInput&, Split, Tran, std::optional<float>);
    void _set_slew(PrimaryInput&, Split, Tran, std::optional<float>);
    void _set_rat(PrimaryOutput&, Split, Tran, std::optional<float>);
    void _set_load(PrimaryOutput&, Split, Tran, std::optional<float>);
    void _cppr(bool);
    void _topologize(SfxtCache&, size_t) const;
    void _spfa(SfxtCache&) const;
    void _spdp(SfxtCache&) const;
    void _recover_prefix(Path&, const SfxtCache&, size_t) const;
    void _recover_datapath(Path&, const SfxtCache&) const;
    void _recover_datapath(Path&, const SfxtCache&, const PfxtNode*, size_t) const;
    void _enable_full_timing_update();
    void _merge_celllib(Celllib&, Split);
    void _insert_full_timing_frontiers();
    void _spur(Endpoint&, size_t, PathHeap&) const;
    void _spur(PfxtCache&, const PfxtNode&) const;
    void _dump_graph(std::ostream&) const;
    void _dump_power(std::ostream&) const;
    void _dump_taskflow(std::ostream&) const;
    void _dump_cell(std::ostream&, const std::string&, Split) const;
    void _dump_celllib(std::ostream&, Split) const;
    void _dump_net_load(std::ostream&) const;
    void _dump_pin_cap(std::ostream&) const;
    void _dump_slew(std::ostream&) const;
    void _dump_slack(std::ostream&) const;
    void _dump_at(std::ostream&) const;
    void _dump_rat(std::ostream&) const;
    void _dump_timer(std::ostream&) const;
    void _dump_timing(std::ostream&) const;
    void _dump_verilog(std::ostream&, const std::string&) const;
    void _dump_spef(std::ostream&) const;
    void _dump_rctree(std::ostream&) const;

    template <typename... T, std::enable_if_t<(sizeof...(T)>1), void>* = nullptr >
    void _insert_frontier(T&&...);
    
    SfxtCache _sfxt_cache(const Endpoint&) const;
    SfxtCache _sfxt_cache(const PrimaryOutput&, Split, Tran) const;
    SfxtCache _sfxt_cache(const Test&, Split, Tran) const;
    CpprCache _cppr_cache(const Test&, Split, Tran) const;
    PfxtCache _pfxt_cache(const SfxtCache&) const;

    Net& _insert_net(const std::string&);
    Pin& _insert_pin(const std::string&);
    Arc& _insert_arc(Pin&, Pin&, Net&);
    Arc& _insert_arc(Pin&, Pin&, Test&);
    Arc& _insert_arc(Pin&, Pin&, TimingView);
    SCC& _insert_scc(std::vector<Pin*>&);
    Test& _insert_test(Arc&);
    Clock& _create_clock(const std::string&, Pin&, float);
    Clock& _create_clock(const std::string&, float);

    std::optional<float> _report_at(const std::string&, Split, Tran);
    std::optional<float> _report_rat(const std::string&, Split, Tran);
    std::optional<float> _report_slew(const std::string&, Split, Tran);
    std::optional<float> _report_slack(const std::string&, Split, Tran);
    std::optional<float> _report_load(const std::string&, Split, Tran);
    std::optional<float> _cppr_credit(const Test&, Split, Tran) const;
    std::optional<float> _cppr_credit(const CpprCache&, Pin&, Split, Tran) const;
    std::optional<float> _cppr_offset(const CpprCache&, Pin&, Split, Tran) const;
    std::optional<float> _sfxt_offset(const SfxtCache&, size_t) const;
    
    size_t _max_pin_name_size() const;
    size_t _max_net_name_size() const;

    inline auto _encode_pin(Pin&, Tran) const;
    inline auto _decode_pin(size_t) const;
    inline auto _encode_arc(Arc&, Tran, Tran) const;
    inline auto _decode_arc(size_t) const;
    inline auto _has_state(int) const;
    inline auto _insert_state(int);
    inline auto _remove_state(int = 0);

    /*
     * Vivek's implementation 
     */

    // runtime
    size_t _initialize_DAG_time = 0;
    size_t _partition_DAG_time = 0;
    size_t _choose_least_cost_task_time = 0;
    size_t _find_merge_candidates_time = 0;
    size_t _try_merging_time = 0;
    size_t _choose_least_cost_neighbor_time = 0; 
    size_t _perform_merge_time = 0; 
    size_t _update_top_down_vector = 0;
    size_t _update_global_task_vector = 0; 
    size_t _vivek_btask_rebuild_time = 0;
    size_t _vivek_btask_runtime = 0;

    size_t GDCA_dfs_time = 0;
    size_t GDCA_build_coarsen_graph_time = 0;

    size_t CPU_topo_runtime = 0;
    size_t GPU_topo_runtime = 0;

    VivekDAG _vivekDAG;
    VivekDAG _rebuild_vivekDAG;   

    // csr format of _vivekDAG
    std::vector<int> _adjp; // edge offset 
    std::vector<int> _adjncy; // flatterned adjacency list
    std::vector<int> _adjncy_size; // number of edges of each node
    std::vector<int> _dep_size; // number of dependents of each node
    std::vector<int> _suc_size; // number of successors of each node
    std::vector<int> _topo_result_cpu;
    std::vector<int> _topo_result_gpu;
    std::vector<int> _partition_result_gpu;
    std::vector<int> _partition_result_cpu;
    std::vector<int> _partition_counter_gpu;
    std::vector<int> _partition_counter_cpu;
    int _total_num_partitions_cpu = 0;
    int _total_num_partitions_gpu = 0;

    std::stack<int> _top_down_topo_order_cur_vivekDAG; // topological order from top to bottom for current DAG
    std::vector<int> _top_down_topo_order_cur_vivekDAG_vector; // topological order from top to bottom for current DAG

    // global priority queue to store all the vivekTask
    std::priority_queue<VivekTask*, std::vector<VivekTask*>, CompareTaskByCost> _global_task_queue;
    std::vector<VivekTask*> _global_task_vector;
    std::vector<VivekTask*> _global_task_vector_GDCA;
    std::priority_queue<VivekTask*, std::vector<VivekTask*>, CompareTaskByCost> _global_task_queue_GDCA;
    std::queue<Pin*> _global_pin_queue_GDCA;

    // initialize local critical path cost for each pins
    void _initialize_local_crit_cost_pins();

    // initialize vivekDAG and all the tasks to global priority queue
    void _initialize_vivekDAG();

    // export csr format of vivekDAG
    void _export_csr();

    // run update_timing sequentially according to topo order result from GPU
    void _run_topo_gpu();

    // partition vivekDAG
    void _partition_vivekDAG();
    void _partition_vivekDAG_GDCA_origin(); // older version of GDCA implementation
    void _partition_vivekDAG_GDCA_cpu();
    void _partition_vivekDAG_GDCA_gpu();
    void _partition_vivekDAG_GDCA_topo();
    void _GDCA_dfs();
    void _GDCA_build_coarsen_graph();  
    void _GDCA_build_coarsen_graph_gpu();
    void _GDCA_build_coarsen_graph_cpu();
    void _GDCA_build_coarsen_graph_topo();
    void _GDCA_build_coarsen_graph_par();

    // print runtime for each task
    void _task_timing_profile();

    // print runtime for each steps in partition
    void _partition_timing_profile(); 

    // calculate maximum parallelism
    void _print_max_parallelism();

    // check if merging vtask1 and vtask2 will lead to cycle in vivekDAG
    bool _form_cycle(VivekTask* vtask1, VivekTask* vtask2);

    // dfs used in _form_cycle
    void _dfs_form_cycle(VivekTask* start, VivekTask* end);

    // merge vtask1 and vtask2 in vivekDAG
    void _try_merging(VivekTask* least_cost_task, std::vector<VivekTask*>& local_task_vector, size_t& merge_cnt);
    void _merge_vivekTasks(VivekTask* vtask1, VivekTask* vtask2, bool neighbor_is_fanout); 

    // insert the merged vtask into the right position of _global_task_vector
    void _insert_merged_vivekTask(VivekTask* merged_vtask);

    // update local critical path cost 
    void _update_local_crit_cost();

    // dfs in update_local_crit_cost() to get topological order of current vivekDAG
    void _dfs_topo_order(int start, std::stack<int>& top_down_result, std::queue<int>& bottom_up_result);

    // print updated cost of tasks in vivekDAG
    void _print_updated_cost();

    // rebuild _taskflow according to vivekDAG
    void _rebuild_taskflow_vivek();
    void _rebuild_taskflow_GDCA();
    void _run_vivekDAG_GDCA_seq(); // run GDCA partitioned vivekDAG sequentially

    // check if DAG has cycle
    bool _check_DAG_cycle();
    bool _isCyclicUtil(VivekTask* vtask);
    int _total_task_visited = 0;

    // rebuild ftask part of _taskflow
    void _rebuild_ftask();

    // reset vivek's partition
    void _reset_partition();
};

// Procedure: _insert_frontier
template <typename... T, std::enable_if_t<(sizeof...(T)>1), void>*>
void Timer::_insert_frontier(T&&... pins) {
  (_insert_frontier(pins), ...);
}
    
// Function: num_primary_inputs
inline auto Timer::num_primary_inputs() const {
  return _pis.size();
}

// Function: num_primary_outputs
inline auto Timer::num_primary_outputs() const {
  return _pos.size();
}

// Function: num_pins
inline auto Timer::num_pins() const {
  return _pins.size();
}

// Function: num_nets
inline auto Timer::num_nets() const {
  return _nets.size();
}

// Function: num_arcs
inline auto Timer::num_arcs() const {
  return _arcs.size();
}

// Function: num_gates
inline auto Timer::num_gates() const {
  return _gates.size();
}

// Function: num_tests
inline auto Timer::num_tests() const {
  return _tests.size();
}

// Function: num_sccs
inline auto Timer::num_sccs() const {
  return _sccs.size();
}
    
// Function: time_unit
inline auto Timer::time_unit() const {
  return _time_unit;
}

// Function: power_unit
inline auto Timer::power_unit() const {
  return _power_unit;
}

// Function: resistance_unit
inline auto Timer::resistance_unit() const {
  return _resistance_unit;
}

// Function: current_unit
inline auto Timer::current_unit() const {
  return _current_unit;
}

// Function: voltage_unit
inline auto Timer::voltage_unit() const {
  return _voltage_unit;
}

inline std::optional<float> Timer::cell_voltage() const {
  int n_volt=0;
  float voltage=0;

  FOR_EACH_EL_IF(el, _celllib[el]) {
    if (_celllib[el]->voltage) {
      n_volt++;
      voltage += _celllib[el]->voltage.value();
    }
  }

  if (n_volt) {
    return {voltage/n_volt};
  }
  return std::nullopt;
}

// Function: capacitance_unit
inline auto Timer::capacitance_unit() const {
  return _capacitance_unit;
}

// Function: primary_inputs
// expose the primary input data structure to users    
inline const auto& Timer::primary_inputs() const {
  return _pis;
}

// Function: primary_outputs
// Expose the primary output data structure to users
inline const auto& Timer::primary_outputs() const {
  return _pos;
}

// Function: pins
// Expose the pin data structure to users
inline const auto& Timer::pins() const {
  return _pins;
}

// Function: nets
// Expose the net data structure to users
inline const auto& Timer::nets() const {
  return _nets;
}

// Function: gates
// Expose the gate data structure to users
inline const auto& Timer::gates() const {
  return _gates;
}

// Function: clocks
// Expose the clock data structure to users
inline const auto& Timer::clocks() const {
  return _clocks;
}

// Function: tests
// Expose the test data structure to users
inline const auto& Timer::tests() const {
  return _tests;
}

// Function: arcs
// Expose the arc data structure to users
inline const auto& Timer::arcs() const {
  return _arcs;
}

// Function: _encode_pin
inline auto Timer::_encode_pin(Pin& pin, Tran rf) const {
  return rf == RISE ? pin._idx : pin._idx + _idx2pin.size();
}

// Function: _decode_pin
inline auto Timer::_decode_pin(size_t idx) const {
  return std::make_tuple(_idx2pin[idx%_idx2pin.size()], idx<_idx2pin.size() ? RISE : FALL);
}

// Function: _encode_arc
inline auto Timer::_encode_arc(Arc& arc, Tran frf, Tran trf) const {
  if(frf == RISE) {
    return arc._idx + (trf == RISE ? 0 : _idx2arc.size());
  }
  else {
    return arc._idx + (trf == RISE ? _idx2arc.size()*2 : _idx2arc.size()*3);
  }
}

// Function: _decode_arc
inline auto Timer::_decode_arc(size_t idx) const {
  if(auto s = _idx2arc.size(); idx < s) {
    return std::make_tuple(_idx2arc[idx % s], RISE, RISE);
  }
  else if(idx < 2*s) {
    return std::make_tuple(_idx2arc[idx % s], RISE, FALL);
  }
  else if(idx < 3*s) {
    return std::make_tuple(_idx2arc[idx % s], FALL, RISE);
  }
  else {
    return std::make_tuple(_idx2arc[idx % s], FALL, FALL);
  }
}

// Function: _has_state
inline auto Timer::_has_state(int s) const {
  return _state & s;
}

// Procedure: _insert_state
inline auto Timer::_insert_state(int s) {
  _state |= s;
}
  
// Procedure: _remove_state
inline auto Timer::_remove_state(int s) {
  if(s == 0) _state = 0;
  else {
    _state &= ~s;
  }
}

};  // end of namespace ot ------------------------------------------------------------------------

#endif




