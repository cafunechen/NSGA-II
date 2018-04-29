%%
% Implementação NSGA-II (Non-dominated Sorting Genetic Algorithm II)
% DE (Differential Evolution)
% Autor: Thiago Silva
%
function NSGAII_DE()
Npop = 100;
TempNpop = Npop*2;
Nvar = 1;
Nobj = 2;
Ngen = 100;
Fator = 0.7;
CR = 0.9;
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

CDist = @(Pop, Npop) CrowdingDistance(Pop, Npop, Nvar, Nobj, xmax, xmin);

% Fitness População
for K = 1:Npop
    Pop(K, Nvar+1:L) = Fitness(Pop(K, :), Nvar);
end

% Gerações
for gen = 1:Ngen
    
    % Construção da próxima geração
    TempPop(1:Npop, :) = Pop;
    
    % Seleção
    F = FastNonDominatedSort(Pop, Npop, Dom);
    for K=1:Npop
        SelIndex = Torneio(F, Pop, Npop, CDist);
        SelPop(K, :) = Pop(SelIndex, :);
    end
    
    % Randomiza seleção
    Rands = randperm(Npop);
    
    % Cruzamento
    K = 1;
    while K <= Npop
        Sigma = randsample(Npop, 2);
        X1 = SelPop(Rands(K), :);
        X2 = SelPop(Rands(mod(Sigma(1), Npop) + 1), :);
        X3 = SelPop(Rands(mod(Sigma(2), Npop) + 1), :);
        V = X1 + Fator.*(X2 - X3);
        Index = randi(Nvar);
        for i=1:Nvar
            if rand() < CR || Index ~= i
                V(i) = X1(i);
            end
        end
        V(Nvar+1:L) = Fitness(V, Nvar);
        
        Temp = [X1; V];
        Ftemp = FastNonDominatedSort(Temp, size(Temp,1), Dom);
        if Torneio(Ftemp, Temp, 2, CDist) == 1
            TempPop(Npop+K, :) = X1;
        else
            TempPop(Npop+K, :) = V;
        end
        
        K = K + 1;
    end
    
    % Mutação
    K = 1;
    while K <= Npop
        if rand() < Pm
            X = SelPop(Rands(K), :);
            X = MutacaoPolinomial(X, Nvar, xmax, xmin);
            X(Nvar+1:L) = Fitness(X, Nvar);
            TempPop(Npop+K, :) = X;
        end
        K = K + 1;
    end
    
    % Atualiza fronteiras Pareto
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
    
    % População com próxima fronteira
    while NpopAtual < Npop
        Temp = TempPop(F == Findex, :);
        Ntemp = size(Temp, 1);
        
        if Ntemp == 1
            Pop(NpopAtual+1, :) = Temp(1, :);
        else
            Sigma = randsample(Ntemp, 2);
            Dist = CDist(Temp, Ntemp);
            if Dist(Sigma(1)) > Dist(Sigma(2))
                Pop(NpopAtual+1, :) = Temp(Sigma(1), :);
            else
                Pop(NpopAtual+1, :) = Temp(Sigma(2), :);
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
function R = Fitness(X, Nvar)
x = X(1:Nvar);
R = zeros(2, 1);
R(1) = x.^2;
R(2) = (x-2).^2;
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
function Dist = CrowdingDistance(P, NP, Nvar, Nobj, xmax, xmin)
Dist = zeros(NP, 1);
for M=1:Nobj
    [~, Pos] = sort(P(:, Nvar+M));
    Dist(Pos(1)) = Inf;
    Dist(Pos(end)) = Inf;
    for K=2:NP-1
        Dist(Pos(K)) = Dist(Pos(K)) + abs((P(Pos(K+1), Nvar+M) - P(Pos(K-1), Nvar+M)) / abs(xmax - xmin));
    end
end
end

%%
% Torneio Binário
function SelIndex = Torneio(F, Pop, Npop, CDist)
Index = randperm(Npop, 2);
if F(Index(1)) < F(Index(2))
    SelIndex = Index(1);
elseif F(Index(1)) > F(Index(2))
    SelIndex = Index(2);
else
    Pos = find(F == F(Index(1)));
    Temp = Pop(Pos, :);
    Ntemp = size(Temp, 1);
    
    Dist = CDist(Temp, Ntemp);
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
% Mutação Polinomial (Deb & Goyal, 1996)*
function M = MutacaoPolinomial(X, Nvar, xmax, xmin)
Eta = 5;
Sigma = 0.4;
V = zeros(1, length(X));

for K = 1:Nvar
    u = rand();
    if u <= 0.5
        Delta = (2*u)^(1/(Eta+1)) - 1;
    else
        Delta = 1 - (2*(1-u))^(1/(Eta+1));
    end
    V(K) = Sigma * (xmax - xmin) * Delta;
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
