
numWorkers=$1
iterNum=$2

# echo "Iter $iterNum for $numWorkers workers"

open_terminal()
{
	script=$1
	if [[ "$OSTYPE" == "darwin"* ]]; then # Running on mac
		osascript <<END 
		tell app "Terminal" to do script "cd \"`pwd`\"; $script; exit"
END
	elif [[ "$OSTYPE" == "linux-gnu" ]]; then # Running on linux
		DISPLAY=:0 xterm -e $script $2
	fi
}

run()
{
	ports=()
	for j in `seq 1 $numWorkers` 
	do
		port=$((4000+$j))
		ports+=($port)
		open_terminal "python3 -B train_controller.py --selfport $port --iternum $iterNum" &
	done

	sleep $numWorkers
	port_string=$( IFS=$' '; echo "${ports[*]}" )
	open_terminal "python3 -B train_controller.py --workerports $port_string --iternum $iterNum"
}

run
