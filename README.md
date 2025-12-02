# sv-fibergen
Python + SVmultiphysics codes for fiber generation

### Notes on SVmultiphysics solver

To solve a Laplace equation directly from the transient HEAT solver in SVmultiphysics, in `<GeneralSimulationParameters>` we need to set,
```
<Number_of_time_steps> 1 </Number_of_time_steps>
<Time_step_size> 1 </Time_step_size>
<Spectral_radius_of_infinite_time_step> 0. </Spectral_radius_of_infinite_time_step>
```
and in `<Add_equation type="heatS" >`,
```
<Conductivity> 1.0 </Conductivity>
<Source_term> 0.0 </Source_term>
<Density> 0.0 </Density>
```
This will allow us to solve the Laplace equation directly in 1 timestep and 1 iteration.