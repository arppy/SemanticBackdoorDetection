#!/bin/bash

if [ $# -ne 1 ]; then
  echo 'usage: '$0' fname';
  exit 1;
fi;

fname=$1
echo $fname;
t=0;
for b in `echo \`basename $fname\` | awk -F '-' '{for(i=2;i<NF;i++){print $i}}'`; do 
  echo "EVAL: backdoor: "$b" target: "$t;
  python model_ASR.py --load $fname --backdoor_class $b --target_class $t
  t=`expr $t + 1`;
done

