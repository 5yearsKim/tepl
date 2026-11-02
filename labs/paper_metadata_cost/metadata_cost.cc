#include <chrono>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <nlohmann/json.hpp>
#include "src/executable/program.h"
#include "src/rewrite/matcher.h"
using namespace tepl;
int main(int argc,char**argv){try{
 if(argc!=2)throw std::runtime_error("expected compiled program");
 std::ifstream f(argv[1]);std::string text{std::istreambuf_iterator<char>(f),{}};
 auto p=executable::load(text);nlohmann::json rows=nlohmann::json::array();
 const auto&r=p.ir().rules.at(0);
 for(const auto&g:p.ir().graphs){
  rewrite::Bindings base;rewrite::Budget initial;
  if(!rewrite::match(p.ir(),r,g,g.root,base,initial))throw std::runtime_error("match failed");
  auto wanted=rewrite::list(rewrite::attributeValue(g.nodes.at(1).attributes.at(0)));
  auto order=rewrite::list(rewrite::attributeValue(g.nodes.at(2).attributes.at(0)));
  rewrite::List expected;for(auto i:order)expected.push_back(wanted.at(rewrite::position(i)));
  auto once=[&](bool check){auto bindings=base;rewrite::Budget budget;rewrite::Evaluator eval(p,g,0,bindings,budget);
   if(!eval.evaluateWhere())throw std::runtime_error("where rejected");
   for(const auto&d:r.descriptors)if(d.kind==core::completed::Descriptor::Kind::kDerived){
    const auto&fields=bindings.descriptors.at(d.id.value).fields;
    if(check && rewrite::list(fields.at(0))!=expected)throw std::runtime_error("wrong permutation");
   }
  };
  once(true);
  for(int round=-1;round<10;++round){auto start=std::chrono::steady_clock::now();
   for(int i=0;i<1000;++i)once(false);
   double ns=std::chrono::duration<double,std::nano>(std::chrono::steady_clock::now()-start).count()/1000;
   rows.push_back({{"graph",g.name},{"rank",expected.size()},{"round",round},{"matches",1000},{"ns_per_match",ns}});
  }
 }
 std::cout<<rows.dump(2)<<"\n";
 }catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
