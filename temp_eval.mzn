var int: a;
constraint a - 10 != 10 - a -> 10 * a <= a * 10;
solve minimize a;
output [show(a)];
