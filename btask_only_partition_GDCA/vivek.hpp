#ifndef OT_TIMER_VIVEK_HPP_
#define OT_TIMER_VIVEK_HPP_

#include <ot/timer/pin.hpp>


namespace ot {

  class Timer;
  class VivekTask;
  class VivekDAG;

  class VivekTask {
    friend class Timer;
    friend class VivekDAG;
    public:
      VivekTask( 
          int id,
          int local_crit_cost,
          std::vector<std::pair<bool, Pin*>>& pins) : _id(id), _local_crit_cost(local_crit_cost), _pins(pins) {} 
          
      VivekTask( 
          int id,
          int local_crit_cost,
          std::pair<bool, Pin*> pin) : _id(id), _local_crit_cost(local_crit_cost) {
        _pins.push_back(pin);
      } 

      void addFanin(int id) {
        _fanin.push_back(id);
      }

      void addFanout(int id) {
        _fanout.push_back(id);
      }

      void deleteRepFan() { // delete replicate fanin/fanout 
                            // only need to be executed once(in initialization)
        std::set<int> fanin_set(_fanin.begin(), _fanin.end());
        std::set<int> fanout_set(_fanout.begin(), _fanout.end());
        _fanin.assign(fanin_set.begin(), fanin_set.end());
        _fanout.assign(fanout_set.begin(), fanout_set.end());
      }

      void getSelfCost() {
        int self_cost = 0;
        for(auto& pair : _pins) {
          if(pair.first) {
            self_cost = self_cost + pair.second->_fself_cost; 
          }
          else {
            self_cost = self_cost + pair.second->_bself_cost; 
          }
        }
        _self_cost = self_cost;
      }

      ~VivekTask() {
        // std::cerr << "destructor called\n";
      }

    private:
      int _id;
      size_t _cluster_id;
      tf::Task _tftask;
      size_t _runtime; 
      int _local_crit_cost;
      int _self_cost;
      int _prev_crit_cost = 0;
      int _after_crit_cost = 0;
      std::vector<std::pair<bool, Pin*>> _pins; // bool = true, forward task
      std::vector<int> _fanin; // int represents vtask's id
      std::vector<int> _fanout;
      size_t _num_deps_release = 0;
      int _num_visited = 0; 
      bool _merged = false; // indicate this task has been merged, 
                            // merged indicates this task is no longer in vivekDAG
      bool _pushed = false;
  };

  class VivekDAG {
    friend class Timer;
    friend class VivekTask;
    public:
      void addVivekTask(
          int id,
          int local_crit_cost,
          std::pair<bool, Pin*> pin) {
        VivekTask* newTask = new VivekTask(id, local_crit_cost, pin);
        _vtask_ptrs.push_back(newTask); // try emplace_back?
        _vtasks.push_back(*newTask);
      }

      void addVivekTask(
          int id,
          int local_crit_cost,
          std::vector<std::pair<bool, Pin*>>& pins) {
        VivekTask* newTask = new VivekTask(id, local_crit_cost, pins);
        _vtask_ptrs.push_back(newTask); // try emplace_back?
        _vtasks.push_back(*newTask);
      }

      void resetVivekDAG() {
        _vtask_ptrs.clear();
        _vtasks.clear();
      }

    private:
//      std::vector<VivekTask> _vtasks;
      std::vector<VivekTask*> _vtask_ptrs;
      std::list<VivekTask> _vtasks;
      std::vector<std::vector<VivekTask*>> _vtask_clusters;
  };
}

#endif
