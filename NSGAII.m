%%
% Implementação NSGA-II (Non-dominated Sorting Genetic Algorithm II)
% Autor: Thiago Silva
%
function NSGAII()
Npop = 100;
TempNpop = Npop*2;
Nvar = 1;
Nobj = 2;
Ngen = 100;
Pc = 0.95;
Pm = 1/Nvar;

% Limites do problema
xmax = 1000;
xmin = -1000;
L = Nvar+Nobj;

% População inicial
Pop = IniPop(Npop, Nvar, L, xmax, xmin);
SelPop = zeros(Npop, L);
TempPop = zeros(TempNpop, L);

% Problema de minimização
Dom = @(a, b) Dominate(a, b, Nvar, Nobj);

% Distância de multidão
CrowDist = @(Pop, Npop) CrowdingDistance(Pop, Npop, Nvar, Nobj, xmax, xmin);

% Fitness População
for k = 1:Npop
    Pop(k, Nvar+1:L) = Fitness(Pop(k, :), Nvar);
end

% Gerações
for gen = 1:Ngen
    
    % Construção da próxima geração
    TempPop(1:Npop, :) = Pop;
    
    % Seleção
    F = FastNonDominatedSort(Pop, Npop, Dom);
    for k = 1:Npop
        SelIndex = Torneio(F, Pop, Npop, CrowDist);
        SelPop(k, :) = Pop(SelIndex, :);
    end
    
    % Randomiza seleção
    Rands = randperm(Npop);
    
    % Cruzamento
    k = 1;
    while k <= Npop
        if rand() < Pc
            Pai1 = SelPop(Rands(k), :);
            Pai2 = SelPop(Rands(mod(k, Npop) + 1), :);
            [Filho1, Filho2] = SBX(Pai1, Pai2, Nvar);
            
            if (Npop-k) > 1
                Filho1(Nvar+1:L) = Fitness(Filho1, Nvar);
                Filho2(Nvar+1:L) = Fitness(Filho2, Nvar);
                TempPop(Npop+k, :) = Filho1;
                TempPop(Npop+k+1, :) = Filho2;
                k = k + 2;
            else
                if rand() < 0.5
                    Filho1(Nvar+1:L) = Fitness(Filho1, Nvar);
                    TempPop(Npop+k, :) = Filho1;
                else
                    Filho2(Nvar+1:L) = Fitness(Filho2, Nvar);
                    TempPop(Npop+k, :) = Filho2;
                end
                k = k + 1;
            end
        end
    end
    
    % Mutação
    k = 1;
    while k <= Npop
        if rand() < Pm
            X = SelPop(Rands(k), :);
            X = MutacaoPolinomial(X, Nvar, xmax, xmin);
            X(Nvar+1:L) = Fitness(X, Nvar);
            TempPop(Npop+k, :) = X;
        end
        k = k + 1;
    end
    
    % Adiciona as fronteiras
    F = FastNonDominatedSort(TempPop, TempNpop, Dom);
    Findex = 1;
    NpopAtual = 0;
    while true
        Temp = TempPop(F == Findex, :);
        Ntemp = size(Temp, 1);
        if (NpopAtual+Ntemp) >= Npop
            break
        end
        Pop(NpopAtual+1:NpopAtual+Ntemp, :) = Temp;
        NpopAtual = NpopAtual + Ntemp;
        Findex = Findex + 1;
    end
    
    % Adiciona indivíduos da próxima fronteira para completar a população
    % na próxima geração
    while NpopAtual < Npop
        Temp = TempPop(F == Findex, :);
        Ntemp = size(Temp, 1);
        if Ntemp == 1
            Pop(NpopAtual+1, :) = Temp(1, :);
        else
            Index = randperm(Ntemp, 2);
            Dist = CrowDist(Temp, Ntemp);
            if Dist(Index(1)) > Dist(Index(2))
                Pop(NpopAtual+1, :) = Temp(Index(1), :);
            else
                Pop(NpopAtual+1, :) = Temp(Index(2), :);
            end
        end
        NpopAtual = NpopAtual + Ntemp;
    end
    
    F = FastNonDominatedSort(Pop, Npop, Dom);
    figure(1)
    plot(Pop(F == 1, Nvar+1), Pop(F == 1, Nvar+2), 'r*')
    grid on
    title(['Iteração ' num2str(gen)])
    pause(0.001)
end
end

%%
% Verifica se o vetor A domina o vetor B.
%  Se R = 1,  então A domina B
%  Se R = -1, então B domina A
%  Senão A e B são incomparáveis
function R = Dominate(A, B, Nvar, Nobj)
I = Nvar+1;
J = Nvar+Nobj;

if all(A(I:J) <= B(I:J)) && any(A(I:J) < B(I:J))
    R = 1;
elseif  all(A(I:J) >= B(I:J)) && any(A(I:J) > B(I:J))
    R = -1;
else
    R = 0;
end
end

%%
% Função-objetivo: 
%   f1(x) = x^2
%   f2(x) = (x-2)^2
function fx = Fitness(X, Nvar)
x = X(1:Nvar);
fx = zeros(2, 1);
fx(1) = x.^2;
fx(2) = (x-2).^2;
end

%%
% Inicaliza população
function P = IniPop(Npop, Nvar, info, xmax, xmin)
P = zeros(Npop, info);

for i = 1:Npop
    for j = 1:Nvar
        P(i, j) = (xmax-xmin)*rand() + xmin;
    end
end
end

%%
% Distância de multidão
function Dist = CrowdingDistance(Pop, NP, Nvar, Nobj, xmax, xmin)
Dist = zeros(NP, 1);

for m = 1:Nobj
    [~, Pos] = sort(Pop(:, Nvar+m), 'descend');
    
    Dist(Pos(1)) = Inf;
    Dist(Pos(end)) = Inf;
    
    for k = 2:NP-1
        Dist(Pos(k)) = Dist(Pos(k)) + abs((Pop(Pos(k+1), Nvar+m) - Pop(Pos(k-1), Nvar+m)) / (xmax - xmin));
    end
end
end

%%
% Torneio Binário
function SelIndex = Torneio(F, Pop, Npop, CrowDist)
Index = randperm(Npop, 2);

if F(Index(1)) < F(Index(2))
    SelIndex = Index(1);
elseif F(Index(1)) > F(Index(2))
    SelIndex = Index(2);
else
    Pos = find(F == F(Index(1)));
    Temp = Pop(Pos, :);
    Ntemp = size(Temp, 1);
    
    Dist = CrowDist(Temp, Ntemp);
    Pa = (Pos == Index(1));
    Pb = (Pos == Index(2));
    
    if Dist(Pa) > Dist(Pb)
        SelIndex = Index(1);
    else
        SelIndex = Index(2);
    end
end
end

%%
% Cruzamento Binário Simulado (SBX) (Deb & Agrawal, 1994)*
function [ F1, F2 ] = SBX(Xi, Xj, Nvar)
F1 = zeros(1, length(Xi));
F2 = zeros(1, length(Xj));
Eta = 2;

for i = 1:Nvar
    u = rand();
    if u <= 0.5
        beta = (2*u) ^ (1/(Eta+1));
    else
        beta = (2*(1-u)) ^ -(1/(Eta+1));
    end
    
    F1(i) = 0.5*((1 + beta)*Xi(i) + (1 - beta)*Xj(i));
    F2(i) = 0.5*((1 - beta)*Xi(i) + (1 + beta)*Xj(i));
end
end

%%
% Mutação Polinomial (Deb & Goyal, 1996)*
function M = MutacaoPolinomial(X, Nvar, xmax, xmin)
Eta = 5;
Sigma = 0.4;
V = zeros(1, length(X));

for k = 1:Nvar
    u = rand();
    if u < 0.5
        Delta = (2*u) ^ (1/(Eta+1)) - 1;
    else
        Delta = 1 - (2*(1-u)) ^ (1/(Eta+1));
    end
    
    V(k) = Sigma * (xmax - xmin) * Delta;
end
M = X + V;
end

%%
% A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II
% Retorna a fronteira Pareto
function F = FastNonDominatedSort(Pop, Npop, Dom)
F = zeros(1, Npop);
S = cell(1, Npop);
N = zeros(1, Npop);

for i = 1:Npop
    p = Pop(i, :);
    for j=1:Npop
        q = Pop(j, :);
        D = Dom(p, q);
        if  D == 1
            S{i} = [S{i} j];
        elseif D == -1;
            N(i) = N(i) + 1;
        end
    end
    if N(i) == 0
        F(i) = 1;
    end
end

i = 1;
FP = find(F == 1);
while ~isempty(FP)
    for p = FP
        for q = S{p}
            N(q) = N(q) - 1;
            if N(q) == 0
                F(q) = i + 1;
            end
        end
    end
    i = i + 1;
    FP = find(F == i);
end
end
