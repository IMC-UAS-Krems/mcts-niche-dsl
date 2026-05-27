var bool: z;
constraint z * z < 100 -> sum(z) > z;
solve satisfy;
output [show(z)];
