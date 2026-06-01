var int: a;
constraint 10;
constraint sum(a) == a;
constraint sum(a) == a;
constraint sum(a) == a \/ a == a;
solve satisfy;
output [show(a)];
