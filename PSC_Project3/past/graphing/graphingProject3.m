clear all; close all; clc;

mag = importdata('./output.csv');
x = linspace (0,1,10);
y = x;
meshgrid(x,y);

contourf(x,y,mag,10)
colorbar