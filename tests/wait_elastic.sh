# https://stackoverflow.com/questions/11904772/how-to-create-a-loop-in-bash-that-is-waiting-for-a-webserver-to-respond/50583452

attempt_counter=0
max_attempts=10

until $(curl --output /dev/null --silent --head --fail http://localhost:9200/); do
    if [ ${attempt_counter} -eq ${max_attempts} ];then
      echo "Max attempts reached, cannot reach elastic"
      exit 1
    fi

    printf '.'
    attempt_counter=$(($attempt_counter+1))
    sleep 5
done
echo "Elastic Ready."
