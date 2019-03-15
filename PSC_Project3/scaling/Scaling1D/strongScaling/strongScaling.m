clear all; close all; clc;
%% Strong Scaling
scaling = importdata('StrongScaling.csv');

nProcs = scaling.data(:,1);
time = scaling.data(:,2);
eff = scaling.data(:,3);

ideal = zeros(numel(nProcs),1);
ideal(:) = 0;
hold on
plot(nProcs,eff,'rs-','LineWidth',2)
plot(nProcs,ideal,'k-','LineWidth',2)
axis([0 70 -10 50])
grid on
grid minor
xlabel('Number of Processes');ylabel('Speed Up');
xticks([4 8 16 32 64])
xticklabels({'4','8','16','32','64'})
movegui('northeast')
legend('Isend & Irecv','Relative','location','NorthEast')