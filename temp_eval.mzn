var bool: z;
constraint z * z < z \/ z;
solve satisfy;
output [show(z)];
