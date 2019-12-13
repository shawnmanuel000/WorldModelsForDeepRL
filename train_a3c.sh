models=([1]="ddpg" [2]="ppo")
iternums=([1]="1") # [2]="0" [3]="-1")

baseport=$1
workers=16
model=$2

open_terminal()
{
	script=$1
	if [[ "$OSTYPE" == "darwin"* ]]; then # Running on mac
		osascript <<END 
		tell app "Terminal" to do script "cd \"`pwd`\"; $script; exit"
END
	elif [[ "$OSTYPE" == "linux-gnu" ]]; then # Running on linux
		xterm -display ":0" -e $script $2 # Add -hold argument after xterm to debug
	fi
}

run()
{
	numWorkers=$1
	runs=$2
	agent=$3
	iterNum=$4

	echo "Workers: $numWorkers, Runs: $runs, Agent: $agent, Iternum: $iterNum"
	
	ports=()
	for j in `seq 1 $numWorkers` 
	do
		port=$((8000+$baseport+$j))
		ports+=($port)
		open_terminal "python3 -B train_a3c.py --selfport $port --iternum $iterNum" &
	done

	sleep 4
	port_string=$( IFS=$' '; echo "${ports[*]}" )
	open_terminal "python3 -B train_a3c.py --runs $runs --model $agent --iternum $iterNum --workerports $port_string"
}

for iter in ${iternums[@]}
do
	runs=250
	if [ $model = "ddpg" ]
	then
		runs=500
	fi 
	run $workers $runs $model $iter
done

