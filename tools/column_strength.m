clear
close
clc

%Specifically written for A7.pdf
%Modifications made to do shear wall design

%To-do: behavior above balance where c > h (approaches infinity)
%       continuing the factored line beyond cap
%       compressive stresses != -60ksi

%Generate constants
%All units in psi, inches


%==========================MAIN==============================
n_step = 1000;

b = 10;
h = 270;
l_w = h;

d = zeros(1,3); %depths are separated to find individual stresses

d(1) = 0.1*l_w;
d(2) = 0.5*l_w;
d(3) = 0.9*l_w;

eps_cu = -3e-3;


f_prime_c = 4000;
f_y = 60000;

E_s = 29000e3;

eps_y = f_y/E_s;

%Calculate areas
Ag = b*h;
Ast = 0.025*Ag; %from ACI 318, App. A, currently No. 10's

As = zeros(1,3); %similar to d
As(1) = 8*.31;
As(2) = 0.0025*0.6*Ag;
As(3) = 8*.31;

%ACI 318-14 Method for ultimate resistance
alpha = 0.85;
beta_1 = 0.85 - (0.05)*(f_prime_c - 4000)/1000;

cap = 0.8;              %0.85 for spiral, 0.8 for tied


%Pure axial resistance
P_o = alpha*f_prime_c*(Ag - Ast) + Ast*f_y;
P_r_o = 0.65*P_o;       %0.65 for tied, 0.75 for spiral
P_r_max = cap*P_r_o;    %0.85 for spiral, 0.8 for tied
P_max = cap*P_o;

%Main curve

%Approach: Given P_n, find the corresponding c, then M_n and phi
P_n = linspace(0,P_o,n_step);
P_r = zeros(1,n_step);
M_n = zeros(1,n_step);
M_r = zeros(1,n_step);
phi = zeros(1,n_step);

for i = 1:n_step
    [M_n(i), phi(i)] = findM(P_n(i),eps_cu,d,eps_y,f_y,...
        E_s,As,f_prime_c,alpha,beta_1,b,h);
    
    M_r(i) = phi(i)*M_n(i);     %factored
    P_r(i) = phi(i)*P_n(i);
    
    if P_r(i) > P_r_max
        P_r(i) = P_r_max;       %capped
    end
    
    if P_n(i) > P_max
        P_n(i) = P_max;
    end
end

%%Complete matrix ends
% P_n = [P_n, P_o];
% P_r = [P_r, P_r_max];
% M_n = [M_n, 0];
% M_r = [M_r, 0];



M_n = M_n/1000/12;              %converted
M_r = M_r/1000/12;
P_n = P_n/1000;
P_r = P_r/1000;

figure
hold on
grid on
plot(M_n,P_n)
plot(M_r,P_r)
title('Moment Interaction Diagram')
xlabel('Moment resistance (k-ft)')
ylabel('Axial resistance (k)')

%========================FUNCTIONS=========================

%Function takes a P_n and finds corresponding c and M_n

function [M,phi] = findM(P,eps_cu,d,eps_y,f_y,E_s,As,f_prime_c,alpha,beta_1,b,h)

%Guessing c
    
    %Start with c and a guess interval. Adjust c based on equilibrium
    %If equil crosses between negative and positive, decrease interval
    %End if error less than 0.01 lbf

%Initiate error handlers
error = 1;
delta_c = 1;
c = 6;
sign_flip = false; %checker on crossing 0 in equilibrium
this_too_much = false; %#ok<*NASGU>
last_too_much = false;

eps_s = zeros(1,3); %Similar to A_s and d
f_s = zeros(1,3);
S_s = zeros(1,3);

while abs(error) > 1e-2
    
    if sign_flip %decrease interval
            delta_c = delta_c/10;
    end
    
    sign_flip = false; %reset checker
    
    for i = 1:3
        
        eps_s(i) = eps_cu*(c - d(i))/c; %strains
        
        if eps_s(i) > eps_y             %check for yielding
            f_s(i) = f_y;
        elseif eps_s(i) < -eps_y
            f_s(i) = -f_y;
        else
            f_s(i) = eps_s(i) * E_s;    %stresses
        end
        
        if f_s(i) > 0
            S_s(i) = As(i)*f_s(i);      %forces
        else
            S_s(i) = As(i)*(-f_s(i) - alpha*f_prime_c);
            S_s(i) = -S_s(i);
        end
    end
    
    Cc = alpha*f_prime_c*beta_1*c*b;    %concrete compression
    
    error = sum(S_s) - Cc + P;  %Here, P is added because a negative
                                %compressive P is subtracted (P is
                                %positive)
    
    %Adjust c, change direction if pass zero
    if error < 0
        c = c - delta_c;
        this_too_much = true;
    else
        c = c + delta_c;
        this_too_much = false;
    end

    %Check for crossing 0 in equilibrium
    if this_too_much ~= last_too_much
        sign_flip = true;
    end

    last_too_much = this_too_much;    
end

phi = 0.65 + (eps_s(3) - eps_y)/(0.005 - eps_y)*0.250;

if phi > 0.9
    phi = 0.9;
elseif phi < 0.65
    phi = 0.65;
end

M = 0;

for j = 1:3
    M = M + abs(S_s(j))*abs(d(j) - h/2);
end

M = M + abs(Cc)*(h/2 - (beta_1*c)/2);

end



