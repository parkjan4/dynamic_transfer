Naming convention:

"MDP policy": Policy from solving uniformized MDP (with (0,0) as an absorbding state) without discount rate using value iteration.
"Fluid policy": Policy from solving fluid control problem at various initial conditions with tau equal to inverse uniformization rate.

"identical": unit holding costs are identical
"non-identical": unit holding costs are non-identical

"light": lambda=(0.55,0.65), mu=(1,1), avg. utilization = 0.6
"moderate": Average utilization fixed at 0.75.
"heavy": lambda=(0.85,0.95), mu=(1,1), avg. utilization = 0.9

"close": The traffic intensities of the queues are "close"
"far": The traffic intensities of the queues are "far apart," i.e., disproportionate dynamics.

"large": transfer costs (m and K) are large.
"small": trnasfer costs are small.

"scaleX": process scaled by a factor of X; relevant only when comparing to fluid policy.
*Fluid policy is not scaled, so doesn't have this extension.