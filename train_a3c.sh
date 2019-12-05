
baseport=$1

open_terminal()
{
	script=$1
	if [[ "$OSTYPE" == "darwin"* ]]; then # Running on mac
		osascript <<END 
		tell app "Terminal" to do script "cd \"`pwd`\"; $script; exit"
END
	elif [[ "$OSTYPE" == "linux-gnu" ]]; then # Running on linux
		xterm -e $script $2
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
		port=$(($baseport+$j))
		ports+=($port)
		open_terminal "python3 train_a3c.py --selfport $port --iternum $iterNum" &
	done

	sleep 4
	port_string=$( IFS=$' '; echo "${ports[*]}" )
	open_terminal "python3 train_a3c.py --runs $runs --model $agent --iternum $iterNum --workerports $port_string"
}

# run 16 500 ddpg -1
# run 16 500 ddpg 0
# run 16 500 ddpg 1

run 16 250 ppo -1
# run 16 250 ppo 0
# run 16 250 ppo 1
