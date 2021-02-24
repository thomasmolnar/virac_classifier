for i in {0..10}; do wget -q https://uhhpc.herts.ac.uk/~lsmith/for_jason/virac_classifier/results/log_$i.log -O log_$i.log; done; python3 ../check_logs.py; head -n 5 summary_log.log;
