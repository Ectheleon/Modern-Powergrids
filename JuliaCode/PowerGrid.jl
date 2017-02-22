#module PowerGrid
#module means you can include("powergrid.jl") and then either
#using PowerGrid # This means it will pull in the exported 'things' into the current scope, you can use myfunction() to call
#import PowerGrid # This means you need to use PowerGrid.myfunction() to call
import MAT # Reading in matlab .mat files
import Images # Doing calculations and transformations on images
import ImageView # Showing images, this package seems to clash with pyplot, so its imported
using Plots # Standard Plotting library with numerous backends (PyPlots,Plotlyjs,Plotly are most used, they may need Pkg.add("...") and switching backends possible with pyplot() or plotlyjs()
using DifferentialEquations # Solving lots of types of DEs numerically.
using LightGraphs # A good graph library, may be useful later.
# Read in data
# Few useful comments:
# vector of vectors != matrix
# map: f.(array) == f applied to every element of array
# splice: f([a,b,c]...) == f(a,b,c)
var=MAT.matread("../Data/ukgrid_data.mat") # Read in data from the working folder (to check: pwd() )
britain=Images.colorview(Images.RGB,Images.normedview(permutedims(var["map"],[3,1,2]))) # Turning the imported picture to a proper object. It is a bit weird as matlabs representation is m by n by 3 and julia needs 3 x m x n (hence permutedims), needs to be normed (normedview) and needs to specify the colorspace (RGB)
pts=[(Int.(var["nodelist"][i,[3,4]])...) for i in indices(var["nodelist"],1)] # Turn the not really nice node list into a nicer form
amat=sparse(Int.(var["A"])) # Turn the dense almost empty adjacency matrix to nice sparse form
edges=[(e...) for e in Set([Set([i,j]) for i in indices(amat,1), j in indices(amat,2) if amat[i,j]!=0])] # Unidirectionality assumed with Set
# Set up sources
pplants=[15,26,33,42,50,51,59,60,61,70,77,79,86,99,108,112,118]
strengths=[2.,3,2,1,1,2,1,4,1,1,2,2,2,2,2,1,5]
pplants=[15,26,33,42,50,51,59,60,61,70,79,86,99,108,112,118]
strengths=[2.,3,2,1,1,2,1,4,1,1,2,2,2,2,1,5]
srcs=zeros(length(pts))
srcs-=sum(strengths)/(length(pts)-length(strengths))
srcs[pplants]=strengths
sum(srcs)
# Plotting Images
canv,img=ImageView.imshow(britain,pixelspacing=[1,1]); # pixelspacing results in the appropriate aspectratio
# Annotate powerplants
annplant=ImageView.AnnotationPoints([p for (i,p) in enumerate(pts) if (i in pplants)],shape='.',size=5,color=Images.RGB(1,0,0)) 
# Annotate consumers
annconsumer=ImageView.AnnotationPoints([p for (i,p) in enumerate(pts) if !(i in pplants)],shape='.',size=3,color=Images.RGB(0,0,1))
# Annotate lines
lines=[(pts[e[1]],pts[e[2]]) for e in edges] # generate all the coordinates for the lines
annlines=ImageView.AnnotationLines(lines,color=Images.RGB(0,0,1)) # Annotation of pictures with blue lines

linehandle=ImageView.annotate!(canv,img,annlines)
phandle=ImageView.annotate!(canv,img,annplant)
chandle=ImageView.annotate!(canv,img,annconsumer)
# ImageView.delete!(canv,pointhandle) # removal of annotations


annplant=ImageView.annotate!(canv,img,ImageView.AnnotationPoints([p for (i,p) in enumerate(pts) if (i in [run[1] for run in result[2]])],shape='.',size=5,color=Images.RGB(0,1,0)))

annlinesb=ImageView.AnnotationLines([(pts[e[1]],pts[e[2]]) for e in [run[1] for run in result[1] if run[2]==false]],color=Images.RGB(1,0,0))
annlinesg=ImageView.AnnotationLines([(pts[e[1]],pts[e[2]]) for e in [run[1] for run in result[1] if run[2]==true]],color=Images.RGB(0,1,0))
linehandleb=ImageView.annotate!(canv,img,annlinesb)
linehandleg=ImageView.annotate!(canv,img,annlinesg)

function resultplot(canv,img,result)
annplantg=ImageView.annotate!(canv,img,ImageView.AnnotationPoints([p for (i,p) in enumerate(pts) if (i in [run[1] for run in result[2] if run[2]==true])],shape='.',size=5,color=Images.RGB(0,1,0)))
annplantb=ImageView.annotate!(canv,img,ImageView.AnnotationPoints([p for (i,p) in enumerate(pts) if (i in [run[1] for run in result[2] if run[2]==false])],shape='.',size=5,color=Images.RGB(1,0,0)))
annlinesb=ImageView.AnnotationLines([(pts[e[1]],pts[e[2]]) for e in [run[1] for run in result[1] if run[2]==false]],color=Images.RGB(1,0,0))
annlinesg=ImageView.AnnotationLines([(pts[e[1]],pts[e[2]]) for e in [run[1] for run in result[1] if run[2]==true]],color=Images.RGB(0,1,0))
linehandleb=ImageView.annotate!(canv,img,annlinesb)
linehandleg=ImageView.annotate!(canv,img,annlinesg)
(annplantb,annplantg,linehandleb,linehandleg)
end

dissipation=1.
coupling=5.

SystemToSolve=Kuramoto(Float64.(amat),srcs,dissipation,coupling,(0.,1,0.),pts,edges)

test(SystemToSolve,[zeros(length(pts)) rand(length(pts))*2π],(0.,50.))
# Graph formulation by LightGraphs
g=Graph()
add_vertices!(g,size(amat,1)) # Add vertices
[add_edge!(g,p...) for p in edges] # Add all edges
f(v)=(v-minimum(v))/(maximum(v)-minimum(v))






vals=global_clustering_coefficient(g)
vals=betweenness_centrality(g)/maximum(betweenness_centrality(g))
pthandles=[ImageView.annotate!(canv,img,ImageView.AnnotationPoint(pts[i],shape='.',size=5,color=weighted_color_mean(vals[i],Colors.RGB(0,0,1),Colors.RGB(1,0,0)))) for i in eachindex(vals)]
[ImageView.delete!(canv,pt) for pt in pthandles]
"""
`functionSetup(couplingMatrix,sources,dissipation,coupling)`

creates the function which is used inside the differential equations solver (compiles it and bakes in all the parameters to machine code)

``\\dot{u}=f(t,u)``
"""
function functionSetup{T}(couplingMatrix::Matrix{T},sources::Vector{T},dissipation::T,coupling::T)
    function f{T}(t::T,u::Matrix{T},du::Matrix{T})
        for i in indices(u,1)
            surge=(( t<15 && t>14.5 && i==1)?15:0)
            #surge=(( t<20 && t>10 && i==1)?0.1:0)
            @inbounds du[i,1]=surge+sources[i]-dissipation*u[i,1]+coupling*sum(couplingMatrix[j,i]*sin(u[j,2]-u[i,2]) for j in indices(couplingMatrix,1) if j!=i)
            @inbounds du[i,2]=u[i,1]
        end
    end
    return f
end

load = 5.06;

prob2=ODEProblem(functionSetup([0.   load    load 0.  0.  0.  0.;
                               load   0.   load   0.  0.  0.  0.;
                               load   load   0. load  0.  0.  0.;
                               0.   0.  load    0.  load  0.  0.;
                               0.   0.  0.  load  0.  load  0.;
                               0.   0.  0.  0.  load  0.  load;
                               0.   0.  0.  0.  0.  load  0.],
                             [1.,1.,1.,2.,0., -3,-2.],1.,1.), # Evolution function f(t,state,derivative)
                  [[0.,0.,0.,0.,0.,0.,0.] [1.,0.,0.,0.,0.,0.,0.]], # Initial matrix
                  (0.,30.)) # Integrate ODE from 0 to 20
#=
prob = ODEProblem(functionSetup([0. 0.5 0.5;
                                0.5 0. 0.4;
                                0. 0.4 0.],
                                [0.2, -0.1, -0.1], 1.,1.), 
                                [[0., 0., 0.] [0.05, 0., 0.]], (0., 20.))
=#

prob=ODEProblem(functionSetup([0.   0.5   0.5;
                               0.5   0.   0.5;
                               0.5   0.5   0.],
                             [0.2, -0.08,-0.12],1.,1.), # Evolution function f(t,state,derivative)
                  [[0.,0.,0.] [1.,0.,0.]], # Initial matrix
                  (0.,30.)) # Integrate ODE from 0 to 20
n=200
randomsystem=rand(n,n)
randomsystem+=randomsystem'
sources=rand(n)
sources[end]=-sum(sources[1:end-1])
prob=ODEProblem(functionSetup(randomsystem,sources,1.,1.),[zeros(n) [1.;zeros(n-1)]],(0.,30.)) # Integrate ODE from 0 to 20

res=solve(prob) # Integrate the problem
toplot=hcat([a[:,2] for a in res[:]]...);
plot(res)
plotlyjs() # Switch backend, as pyplot seems to crash for me for some reason. Another advantage is that this is interactive.
plot(res) # Plot results
#end # close module
