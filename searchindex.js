Search.setIndex({docnames:["getting_started/installing","index","narla/environments","narla/history","narla/multi_agent_network","narla/neurons","narla/rewards","narla/runner","narla/settings","welcome"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":4,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":3,"sphinx.domains.rst":2,"sphinx.domains.std":2,"sphinx.ext.intersphinx":1,"sphinx.ext.todo":2,"sphinx.ext.viewcode":1,nbsphinx:4,sphinx:56},filenames:["getting_started/installing.rst","index.rst","narla/environments.rst","narla/history.rst","narla/multi_agent_network.rst","narla/neurons.rst","narla/rewards.rst","narla/runner.rst","narla/settings.rst","welcome.rst"],objects:{"narla.environments":[[2,0,1,"","ActionSpace"],[2,3,1,"","AvailableEnvironments"],[2,0,1,"","Environment"],[2,0,1,"","GymEnvironment"],[2,3,1,"","GymEnvironments"]],"narla.environments.ActionSpace":[[2,1,1,"","number_of_actions"],[2,2,1,"","sample"],[2,1,1,"","shape"]],"narla.environments.Environment":[[2,1,1,"","action_space"],[2,1,1,"","episode_reward"],[2,2,1,"","has_been_solved"],[2,1,1,"","observation_size"],[2,2,1,"","reset"],[2,2,1,"","step"]],"narla.environments.GymEnvironment":[[2,2,1,"","has_been_solved"],[2,1,1,"","observation_size"],[2,2,1,"","reset"],[2,2,1,"","step"]],"narla.environments.GymEnvironments":[[2,4,1,"","CART_POLE"],[2,4,1,"","MOUNTAIN_CAR"]],"narla.history":[[3,0,1,"","History"]],"narla.history.History":[[3,2,1,"","clear"],[3,2,1,"","get"],[3,2,1,"","record"],[3,2,1,"","sample"],[3,2,1,"","stack"],[3,2,1,"","to_data_frame"]],"narla.multi_agent_network":[[4,0,1,"","Layer"],[4,0,1,"","LayerSettings"],[4,0,1,"","MultiAgentNetwork"],[4,0,1,"","MultiAgentNetworkSettings"]],"narla.multi_agent_network.Layer":[[4,2,1,"","act"],[4,2,1,"","build_connectivity"],[4,2,1,"","distribute_to_neurons"],[4,1,1,"","layer_output"],[4,2,1,"","learn"],[4,1,1,"","neurons"],[4,1,1,"","number_of_neurons"]],"narla.multi_agent_network.LayerSettings":[[4,4,1,"","neuron_settings"],[4,4,1,"","number_of_neurons_per_layer"]],"narla.multi_agent_network.MultiAgentNetwork":[[4,2,1,"","act"],[4,2,1,"","compute_biological_rewards"],[4,2,1,"","distribute_to_layers"],[4,1,1,"","history"],[4,1,1,"","layers"],[4,2,1,"","learn"],[4,2,1,"","record"]],"narla.multi_agent_network.MultiAgentNetworkSettings":[[4,4,1,"","layer_settings"],[4,4,1,"","local_connectivity"],[4,4,1,"","number_of_layers"],[4,4,1,"","reward_types"]],"narla.neurons":[[5,0,1,"","Network"],[5,0,1,"","Neuron"],[5,0,1,"","NeuronSettings"],[5,0,1,"","NeuronTypes"]],"narla.neurons.Network":[[5,2,1,"","forward"],[5,4,1,"","training"]],"narla.neurons.Neuron":[[5,2,1,"","act"],[5,1,1,"","history"],[5,2,1,"","learn"],[5,2,1,"","record"]],"narla.neurons.NeuronSettings":[[5,2,1,"","create_neuron"],[5,4,1,"","learning_rate"],[5,4,1,"","neuron_type"]],"narla.neurons.NeuronTypes":[[5,4,1,"","ACTOR_CRITIC"],[5,4,1,"","DEEP_Q"],[5,4,1,"","POLICY_GRADIENT"],[5,2,1,"","to_neuron_type"]],"narla.neurons.actor_critic":[[5,0,1,"","Network"],[5,0,1,"","Neuron"]],"narla.neurons.actor_critic.Network":[[5,2,1,"","forward"],[5,4,1,"","training"]],"narla.neurons.actor_critic.Neuron":[[5,2,1,"","act"],[5,2,1,"","get_returns"],[5,2,1,"","learn"]],"narla.neurons.deep_q":[[5,0,1,"","Network"],[5,0,1,"","Neuron"]],"narla.neurons.deep_q.Network":[[5,2,1,"","clone"],[5,2,1,"","forward"],[5,4,1,"","training"]],"narla.neurons.deep_q.Neuron":[[5,2,1,"","act"],[5,2,1,"","learn"],[5,2,1,"","sample_history"],[5,2,1,"","update_target_network"]],"narla.neurons.policy_gradient":[[5,0,1,"","Network"],[5,0,1,"","Neuron"]],"narla.neurons.policy_gradient.Network":[[5,2,1,"","clone"],[5,2,1,"","forward"],[5,4,1,"","training"]],"narla.neurons.policy_gradient.Neuron":[[5,2,1,"","act"],[5,2,1,"","get_returns"],[5,2,1,"","learn"]],"narla.rewards":[[6,0,1,"","ActiveNeurons"],[6,0,1,"","BiologicalReward"],[6,0,1,"","LayerSparsity"],[6,0,1,"","Reward"],[6,0,1,"","RewardTypes"]],"narla.rewards.ActiveNeurons":[[6,2,1,"","compute"]],"narla.rewards.BiologicalReward":[[6,2,1,"","compute"]],"narla.rewards.LayerSparsity":[[6,2,1,"","compute"]],"narla.rewards.Reward":[[6,2,1,"","compute"]],"narla.rewards.RewardTypes":[[6,4,1,"","ACTIVE_NEURONS"],[6,4,1,"","LAYER_SPARSITY"],[6,4,1,"","TASK_REWARD"],[6,2,1,"","biological_reward_types"],[6,2,1,"","to_reward"]],"narla.runner":[[7,0,1,"","Job"],[7,0,1,"","Runner"],[7,0,1,"","RunnerSettings"]],"narla.runner.Job":[[7,2,1,"","is_done"]],"narla.runner.Runner":[[7,2,1,"","execute"],[7,2,1,"","is_done"]],"narla.runner.RunnerSettings":[[7,4,1,"","environments"],[7,4,1,"","gpus"],[7,4,1,"","jobs_per_gpu"],[7,4,1,"","learning_rates"],[7,4,1,"","neuron_types"],[7,4,1,"","number_of_layers"],[7,4,1,"","number_of_neurons_per_layer"],[7,2,1,"","product"],[7,4,1,"","reward_types"],[7,4,1,"","settings"]],"narla.settings":[[8,0,1,"","Settings"],[8,0,1,"","TrialSettings"]],"narla.settings.Settings":[[8,4,1,"","environment_settings"],[8,4,1,"","multi_agent_network_settings"],[8,4,1,"","trial_settings"]],"narla.settings.TrialSettings":[[8,4,1,"","batch_size"],[8,4,1,"","device"],[8,4,1,"","gpu"],[8,4,1,"","maximum_episodes"],[8,4,1,"","random_seed"],[8,4,1,"","results_directory"],[8,4,1,"","save_every"],[8,4,1,"","trial_id"]]},objnames:{"0":["py","class","Python class"],"1":["py","property","Python property"],"2":["py","method","Python method"],"3":["py","enum","Python enum"],"4":["py","attribute","Python attribute"]},objtypes:{"0":"py:class","1":"py:property","2":"py:method","3":"py:enum","4":"py:attribute"},terms:{"0":[5,6,7,8],"0001":[5,7],"01":5,"1":[3,7],"10":7,"1000":8,"10000":[3,8],"128":[5,8],"15":[4,7],"2":[6,7],"3":4,"8":0,"abstract":[2,5,6],"class":[2,3,4,5,6,7,8],"enum":[2,5,6],"float":[2,5,7],"function":5,"int":[2,3,4,5,6,7,8],"return":[2,3,4,5,6,7],"static":[4,6],"true":[2,3,4,7],"while":5,A:[4,7],If:[2,3,4,7],The:5,Will:7,access:[2,3,4,5],across:7,act:[4,5],action:[2,4,5],action_spac:2,activ:6,active_neuron:6,actor_crit:7,actorcrit:5,advanc:2,afterward:5,agent:0,alia:0,all:[3,5,7],all_set:7,although:5,alwai:[4,7],an:[2,3,4,5,6,7],append:3,appropri:6,apt:0,ar:[2,6],arbitrari:3,arg:6,argument:[3,4,5],avail:[2,4,5,7],available_environ:7,available_gpu:7,base:[2,3,4,5,6,7,8],base_set:[4,5,7,8],baseset:[4,5,7,8],bashrc:0,batch:8,batch_siz:8,becom:[6,7],been:2,being:8,bin:0,biolog:6,biological_reward:6,biological_reward_typ:6,biologicalreward:4,bool:[2,3,4,5,7],build:4,build_connect:4,call:5,care:5,cart_pol:[2,7],cartpol:[2,7],cd:0,check:2,clear:3,clone:[0,5],column:3,com:0,complet:7,comput:[4,5,6],compute_biological_reward:4,connect:4,contain:[3,4],convert:[3,5,6],correspond:8,cpu:8,creat:[0,3,7],create_neuron:5,cuda:8,current:2,data:[3,4,5,8],datafram:3,deep_q:7,deepq:5,defin:5,desired_spars:6,devic:[7,8],diagon:4,dim:3,dimens:3,distribut:[4,7],distribute_to_lay:4,distribute_to_neuron:4,doesn:3,download:0,dure:[7,8],dynam:6,e:0,each:7,echo:0,element:3,embedding_s:5,enumer:[2,5,6],environ:[0,1,4,5,7,8],environment_set:8,environmentset:8,episod:[2,8],episode_reward:2,evenli:7,everi:[5,8],exampl:5,execut:[4,7],exist:3,factori:[4,7,8],fals:[2,3],follow:2,former:5,forward:5,from:[2,3,4,5],from_most_rec:3,get:[0,2,3,6],get_return:5,git:0,github:0,gpu:[7,8],group:7,gymenviron:7,gymnasium:2,ha:[2,4,7],has_been_solv:2,have:7,histor:2,histori:[1,4,5],hook:5,id:[7,8],ignor:5,index:6,individu:7,inform:5,input:4,input_s:5,instal:1,instanc:5,instead:5,intern:[3,5],is_don:7,jobs_per_gpu:7,kei:[3,4],keyword:[3,5],kwarg:[3,4,5,6],last:[4,7],latter:5,layer:[6,7],layer_index:6,layer_output:4,layer_set:4,layer_spars:6,learn:[4,5,7],learning_r:[5,7],list:[2,3,4,6,7],liter:8,local:[4,5],local_connect:4,m:0,main:7,matrix:4,max:3,maximum_episod:8,mkdir:0,modul:5,mountain_car:2,mountaincar:2,multi:0,multi_agent_network:[1,8],multi_agent_network_set:8,multiagentnetwork:[6,7],multiagentnetworkset:8,n:8,name:[2,3],narla:0,nearbi:4,need:[0,5],network:[0,4,6,7,8],network_set:4,neuron:[4,6,7],neuron_set:4,neuron_typ:[5,7],neuronset:4,neurontyp:7,nn:5,number:[2,4,5,7,8],number_of_act:[2,4,5],number_of_lay:[4,7],number_of_neuron:4,number_of_neurons_per_lay:[4,7],object:[2,3,4,5,6,7],observ:[2,4,5],observation_s:[2,4,5],offset:4,one:[2,4,5,7],onli:[4,7],output:4,output_s:5,overridden:5,p:0,packag:0,paramet:[2,3,4,5,6,7],pass:[5,7],past:2,path:8,pep517:0,per:[4,7],perform:5,phase:4,pip:0,policygradi:5,popen:7,process:7,produc:2,product:7,project:0,properti:[2,4,5],put:[7,8],py:7,python3:0,python_environ:0,pytorch:5,random:8,random_se:8,rang:7,rate:[5,7],receiv:[4,5],recip:5,record:[3,4,5],regist:5,render:2,reset:2,result:8,results_directori:8,reward:[1,2,4,7],reward_typ:[4,5,7],rewardtyp:[4,5],run:[5,7,8],runner:1,s:[2,4,5],sampl:[2,3],sample_histori:5,sample_s:3,save:8,save_everi:8,seed:8,seri:3,set:[1,4,5,7],shape:2,should:5,silent:5,sinc:5,singl:2,site:0,size:[2,3,4,5,8],solv:2,sourc:[2,3,4,5,6,7,8],space:2,sparsiti:6,specifi:6,stack:3,step:[2,8],storag:3,storage_s:3,store:[3,4,5],str:[3,6,8],subclass:5,sudo:0,support:0,system:0,t:3,take:[2,4,5],taken:2,task_reward:6,tensor:[2,3,4,5,6],termin:2,thei:7,them:5,thi:5,time:2,to_data_fram:3,to_neuron_typ:5,to_reward:6,torch:[3,4,5],total:[2,4,7,8],train:[3,5,7,8],trial:8,trial_id:8,trial_set:8,tupl:[2,5],type:[2,3,4,5,6,7],uniqu:8,updat:5,update_target_network:5,us:[0,3,4,5,7,8],v0:2,v1:[2,7],valid:2,valu:[2,3,5,6,7],venv:0,virtual:0,visual:2,what:7,which:[4,5],within:[5,6],word:[3,4],wrapper:2,x:5,yet:3,you:0},titles:["Installing","NaRLA: Neurons as Reinforcement Learning Agents","narla.environments","narla.history","narla.multi_agent_network","narla.neurons","narla.rewards","narla.runner","narla.settings","NaRLA: Neurons as Reinforcement Learning Agents"],titleterms:{actionspac:2,activeneuron:6,actor_crit:5,agent:[1,9],availableenviron:2,biologicalreward:6,deep_q:5,document:1,environ:2,get:1,gymenviron:2,histori:3,instal:0,job:7,layer:4,layerset:4,layerspars:6,learn:[1,9],multi_agent_network:4,multiagentnetwork:4,multiagentnetworkset:4,narla:[1,2,3,4,5,6,7,8,9],network:5,neuron:[1,5,9],neuronset:5,neurontyp:5,policy_gradi:5,reinforc:[1,9],reward:6,rewardtyp:6,runner:7,runnerset:7,set:8,start:1,trialset:8}})