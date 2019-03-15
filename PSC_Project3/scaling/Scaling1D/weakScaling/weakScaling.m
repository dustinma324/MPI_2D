clear all; close all; clc;
%% Weak Scaling
scaling = importdata('WeakScalingMPI.csv');

nProcs = scaling.data(:,1);
time = scaling.data(:,2);
eff = scaling.data(:,3);

ideal = zeros(numel(nProcs),1);
ideal(:) = 100;
hold on
plot(nProcs,eff,'rs-','LineWidth',2)
plot(nProcs,ideal,'k-','LineWidth',2)
axis([0 70 0 120])
grid on
grid minor
xlabel('Number of Processes');ylabel('Cluster Efficiency');
xticks([4 8 16 32 64])
xticklabels({'4','8','16','32','64'})
ytickformat('percentage')
movegui('northeast')
legend('Isend & Irecv','ideal','location','NorthEast')