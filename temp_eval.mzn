var bool: x;
constraint 5 > x + x;
solve minimize x;
output [show(x)];
