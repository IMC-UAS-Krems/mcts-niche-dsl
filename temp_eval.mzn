var bool: c;
var bool: a;
constraint a \/ c in a;
constraint 10 \/ 3;
solve satisfy;
output a;
