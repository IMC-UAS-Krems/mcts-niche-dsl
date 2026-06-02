var set of 10..10: a;
constraint 10 <= a -> 10 mod a;
solve minimize a;
output [show(a)];
