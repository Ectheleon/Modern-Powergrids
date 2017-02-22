module PowerGrid

using DifferentialEquations
using Distributions
using Plots
using ProgressMeter

type Kuramoto{T<:Real,IN<:Integer} <: AbstractParameterizedFunction
    couplingMatrix::AbstractMatrix{T}
    sources::AbstractVector{T}
    dissipation::T
    coupling::T
    peak::Tuple{T,IN,T}
    coords::Vector{Tuple{IN,IN}}
    edges::Vector{Tuple{IN,IN}} end

function (s::Kuramoto{T,IN}){T,IN}(t::T,u::Matrix{T},du::Matrix{T})
    @inbounds for i in indices(u,1)
    surge=((s.peak[1]>t && i==s.peak[2])?s.peak[3]:0.)
    @inbounds du[i,1]=surge+s.sources[i]-s.dissipation*u[i,1]+s.coupling*sum(s.couplingMatrix[j,i]*sin(u[j,2]-u[i,2]) for j in indices(s.couplingMatrix,1) if j!=i)
    @inbounds du[i,2]=u[i,1]
    end end

checkEquilibrium(t,u,integrator;tol=1e-2) = t>5 ? ( norm(u[:,1],Inf)<tol ): false

function runeq(s,ic,time)
    #println("Solve equation")
    cb=DiscreteCallback(checkEquilibrium,(integrator)->terminate!(integrator),(true,false))
    prob=ODEProblem(s,ic,time)
    res=solve(prob,callback=cb)
    return ( res.t[end]<time[2] ? (true,res) : (false,res) )
    end

function test(s,ic,time)
    #val=false
    #while !val
        val,res=runeq(s,ic,time)
    #    if !val println("No equilibrium :(") end
    #end
    #(edgetests(s,res[end],time),peaktests(s,res[end],time))
    if val return (edgetests(s,res[end],time),peaktests(s,res[end],time)) end
    return ([false,res],[false,res])
    end

function addpeak(s,dt,pt,P)
    sout=deepcopy(s)
    sout.peak=(dt,pt,P)
    return sout end

function removeedge(s,e)
    sout=deepcopy(s)
    sout.couplingMatrix[e...]=0
    sout.couplingMatrix[reverse(e)...]=0
    return sout end

function peaktests(s,ic,time;ratio=1.,P=100.,dt=0.5)
    n=Int(ceil(size(s.couplingMatrix,1)*ratio))
    @showprogress "Peak testing: " [(p,runeq(addpeak(s,dt,p,P),ic,time)...) for p in sample(1:size(s.couplingMatrix,1),n,replace=false)]
end

function edgetests(s,ic,time;ratio=1.)
    n=Int(ceil(length(s.edges)*ratio))
    @showprogress "Edge testing: " [(e,runeq(removeedge(s,e),ic,time)...) for e in sample(s.edges,n,replace=false)]
end

end