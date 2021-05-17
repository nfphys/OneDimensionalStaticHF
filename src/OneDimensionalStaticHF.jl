module OneDimensionalStaticHF

greet() = print("Hello World!")

using Plots
using LinearAlgebra
using Parameters
using Arpack
using MyLibrary


"""
    struct PhysicalParam @deftype Float64

mc²: proton mass [MeV]

ħc : Planck constant ħ times speed of light c [MeV fm]

t₀: Skyrme parameter [MeV fm⁻³]

t₃: Skyrme parameter [MeV fm⁻⁶]

a: range of Yukawa potential [fm]

V₀: strength of Yukawa potential [MeV]

ρ₀: nuclear saturation density [fm⁻³]

σ: areal number density of slab [fm⁻²]

Δz: lattice spacing [fm]

Nz: number of grids

zs: array of grids [fm]

cnvl_coeff: convolution coefficients for Yukawa potential
"""
@with_kw struct PhysicalParam{T} @deftype Float64
    mc² = 938.
    ħc  = 197. 
    
    t₀ = -497.726 
    t₃ = 17_270.
    
    a  = 0.45979 
    V₀ = -166.9239/a 
    
    ρ₀ = 0.16 
    σ 

    Δz = 0.1
    Nz::Int64 = 100

    zs::T = range(Δz/2, (Nz-1/2)*Δz, length=Nz)
    
    cnvl_coeff::Vector{Float64} = calc_cnvl_coeff(Δz, a)
end


"""
    calc_cnvl_coeff(param, zs; qmax=25)

Calculate convolution coefficients for Yukawa potential.

    param: My.PhysicalParam
    zs: array of grids in z direction
"""
function calc_cnvl_coeff(Δz, a)
    Y = Δz/a
    qmax = ceil(Int, 15/Y)
    #@show qmax
    
    B₀ = (-4/Y^2       + 3/2)*sinh(Y/2) +       (2/Y)*cosh(Y/2)
    Bp = ( 2/Y^2 + 1/Y + 1/4)*sinh(Y/2) - (1/Y + 1/2)*cosh(Y/2)
    Bm = ( 2/Y^2 - 1/Y + 1/4)*sinh(Y/2) - (1/Y - 1/2)*cosh(Y/2)
    Q  = ( 2/Y^2 + 1/Y + 1/4)*exp(-Y/2) -  2/Y^2
    
    cnvl_coeff = zeros(Float64, qmax+1) # convolution coefficient
    cnvl_coeff[1+0] = 2a*(Q + exp(-Y)*Bm + 1 - exp(-Y/2))
    cnvl_coeff[1+1] = a*exp(-Y)*(B₀ + exp(-Y)*Bm - Q*exp(Y))
    for q in 2:qmax
        cnvl_coeff[1+q] = a*exp(-q*Y)*(B₀ + exp(Y)*Bp + exp(-Y)*Bm)
    end
    
    return cnvl_coeff
end
export calc_cnvl_coeff



param = PhysicalParam(σ=1.4)
@unpack zs = param


function initial_density(param)
    @unpack a, ρ₀, σ, zs = param
    L = σ/ρ₀
    Δz = zs[2] - zs[1]
    
    ρ = similar(zs)
    @. ρ = ρ₀/(1 + exp((zs - 0.5L)/a))
    
    return ρ
end

function test_initial_density(;σ=1.4)
    param = PhysicalParam(σ=σ)
    @time ρ = initial_density(param)
    plot(zs, ρ)
end



function calc_potential!(vpot, param, ρ)
    @unpack mc², ħc, t₀, t₃, a, V₀, Nz, zs, cnvl_coeff = param
    
    fill!(vpot, 0)
    # t₀ term
    @. vpot += (3/4)*t₀*ρ
    
    # t₃ term
    @. vpot += (3/16)*t₃*ρ*ρ
    
    # yukawa term
    qmax = length(cnvl_coeff) - 1
    temp = 2π*a*a*V₀
    for iz in 1:Nz, q in 0:qmax
        if iz+q ≤ Nz
            vpot[iz] += temp*cnvl_coeff[1+q]*ρ[iz+q]
        end
        if iz-q ≥ 1
            vpot[iz] += temp*cnvl_coeff[1+q]*ρ[iz-q]
        elseif 1-(iz-q) ≤ Nz
            vpot[iz] += temp*cnvl_coeff[1+q]*ρ[1-(iz-q)]
        end
    end
    
    # normalization
    @. vpot *= (2mc²/ħc^2)
    
    return vpot
end

function test_calc_potential(;σ=1.4)
    param = PhysicalParam(σ=1.4)
    @unpack zs = param
    
    ρ = initial_density(param)
        
    vpot = similar(zs)
    @time calc_potential!(vpot, param, ρ)
        
    plot(zs, vpot)
end



function make_Hamiltonian(zs, vpot, Π)
    Nz = length(zs)
    Δz = zs[2]-zs[1]
    
    dv = similar(zs)
    @. dv = 2/Δz^2 + vpot
    dv[1] += (-1/Δz^2) * Π
    
    ev = fill(-1/Δz^2, Nz-1)
    
    return SymTridiagonal(dv, ev)
end



function test_make_Hamiltonian(param, zs, Π)
    vpot = @. zs^2
    Hmat = make_Hamiltonian(zs, vpot, Π)
    
    vals, vecs = eigen(Hmat)
    vals[1:10] ./ 2
end




function solve_Hamiltonian(param, zs, ρ; Emax=0, nstates_max=100)
    @unpack ħc, mc² = param
    Nz = length(zs)
    Δz = zs[2] - zs[1]
    
    ψs = zeros(ComplexF64, Nz, nstates_max) # wave function
    spEs = zeros(Float64, nstates_max) # single particle energy
    Πs = Vector{Int64}(undef, nstates_max) 
    
    vpot = similar(zs)
    istate = 0
    for Π in [1, -1]
        calc_potential!(vpot, param, zs, ρ)
        Hmat = make_Hamiltonian(zs, vpot, Π)
        
        vals, vecs = eigen(Hmat)
        
        # normalization
        @. vals *= ħc^2/2mc²
        @. vecs /= sqrt(2Δz)
        
        N = length(vals[vals .< Emax])
        @views for i in 1:N
            if real(vals[i]) > Emax
                continue
            end
            istate += 1
            ψs[:,istate] = vecs[:,i]
            spEs[istate] = real(vals[i])
            Πs[istate] = Π
        end
    end
    
    return ψs[:,1:istate], spEs[1:istate], Πs[1:istate]
end

function solve_Hamiltonian2(param, zs, ρ; Emax=0, nev=20, nstates_max=100)
    @unpack ħc, mc² = param
    Nz = length(zs)
    Δz = zs[2] - zs[1]
    
    ψs = zeros(ComplexF64, Nz, nstates_max) # wave function
    spEs = zeros(Float64, nstates_max) # single particle energy
    Πs = Vector{Int64}(undef, nstates_max) 
    
    vpot = similar(zs)
    istate = 0
    for Π in [1, -1]
        calc_potential!(vpot, param, zs, ρ)
        Hmat = make_Hamiltonian(zs, vpot, Π)
        
        vals, vecs = eigs(Hmat, nev=20, which=:SM)
        
        # normalization
        @. vals *= ħc^2/2mc²
        @. vecs /= sqrt(2Δz)
        
        @views for i in 1:nev
            if real(vals[i]) > Emax
                continue
            end
            istate += 1
            ψs[:,istate] = vecs[:,i]
            spEs[istate] = real(vals[i])
            Πs[istate] = Π
        end
    end
    
    return ψs[:,1:istate], spEs[1:istate], Πs[1:istate]
end

function sort_states(ψs, spEs, Πs)
    p = sortperm(spEs)
    return ψs[:,p], spEs[p], Πs[p]
end

function show_states(ψs, spEs, Πs)
    for i in 1:size(ψs, 2)
        println("i = $i: ")
        @show spEs[i] Πs[i]
        println("")
    end
end

function test_solve_Hamiltonian(param, zs)
    ρ = initial_density(param, zs)
    @time ψs, spEs, Πs = solve_Hamiltonian(param, zs, ρ)
    @time ψs, spEs, Πs = sort_states(ψs, spEs, Πs)
    show_states(ψs, spEs, Πs)
    
    p = plot()
    for i in 1:size(ψs, 2)
        plot!(p, zs, @views @. abs2(ψs[:,i]))
    end
    display(p)
    
    return
end




function func_fermi_energy(Efermi, param, spEs)
    @unpack mc², ħc, σ = param
    
    eq = 0.0
    for i in 1:length(spEs)
        if spEs[i] ≤ Efermi
            eq += 2mc²/(π*ħc*ħc) * (Efermi - spEs[i])
        end
    end
    eq -= σ
end

function calc_fermi_energy(param, spEs; ΔE=0.5)
    Erange = -50:ΔE:0
    
    Efermi = 0.0
    found = false
    for i in 1:length(Erange)-1
        f₁ = func_fermi_energy(Erange[i], param, spEs)
        f₂ = func_fermi_energy(Erange[i+1], param, spEs)
        if f₁*f₂ < 0
            Efermi = bisect(func_fermi_energy, Erange[i], Erange[i+1], 
                args=(param, spEs))
            found = true
            break
        end
    end
    
    if !found
        error("fermi energy not found")
    end
    return Efermi
end

function test_calc_fermi_energy(param, zs)
    ρ = initial_density(param, zs)
    ψs, spEs, Πs = solve_Hamiltonian(param, zs, ρ)
    ψs, spEs, Πs = sort_states(ψs, spEs, Πs)
    
    @time Efermi = calc_fermi_energy(param, spEs)
    @show Efermi func_fermi_energy(Efermi, param, spEs)
    show_states(ψs, spEs, Πs)
end
            



function first_deriv!(dψ, zs, ψ, Π)
    Nz = length(zs)
    Δz = zs[2] - zs[1]
    
    #dψ[1] = (1-Π)*ψ[2]/2Δz
    dψ[1] = (ψ[2] - Π*ψ[1])/2Δz
    for iz in 2:Nz-1
        dψ[iz] = (ψ[iz+1] - ψ[iz-1])/2Δz
    end
    dψ[Nz] = -ψ[Nz-1]/2Δz
    
    return
end

function test_first_deriv!(zs)
    ψ = @. exp(-0.5zs*zs)
    
    dψ = similar(zs)
    first_deriv!(dψ, zs, ψ, 1)
    
    dψ_exact = @. -zs*exp(-0.5zs*zs)
    
    plot(zs, dψ)
    plot!(zs, dψ_exact)
end



function calc_density!(ρ, τ, param, zs, ψs, spEs, Πs, Efermi)
    @unpack mc², ħc = param
    nstates = size(ψs, 2)
    
    fill!(ρ, 0)
    fill!(τ, 0)
    dψ = similar(zs)
    for i in 1:nstates
        if spEs[i] > Efermi
            continue
        end
        σᵢ = (2mc²/(π*ħc*ħc))*(Efermi-spEs[i])
        @views ψ = ψs[:,i]
        first_deriv!(dψ, zs, ψ, Πs[i])
        @. ρ += σᵢ*real(dot(ψ, ψ))
        @. τ += σᵢ*real(dot(dψ, dψ))
        @. τ += (π/2)*σᵢ^2*real(dot(ψ, ψ))
    end
end

function test_calc_density!(param, zs)
    ρ = initial_density(param, zs)
    p = plot()
    plot!(p, zs, ρ; label="ρ₀")
    
    ψs, spEs, Πs = solve_Hamiltonian(param, zs, ρ)
    ψs, spEs, Πs = sort_states(ψs, spEs, Πs)
    
    Efermi = calc_fermi_energy(param, spEs)
    
    τ = similar(zs)
    @time calc_density!(ρ, τ, param, zs, ψs, spEs, Πs, Efermi)
    plot!(p, zs, ρ; label="ρ")
    plot!(p, zs, τ; label="τ")
    display(p)
    
    Δz = zs[2] - zs[1]
    @show sum(ρ)*2Δz
    
    return
end
    



function calc_total_energy(param, zs, ρ, τ)
    @unpack mc², ħc, t₀, t₃, a, V₀, cnvl_coeff = param
    Nz = length(zs)
    Δz = zs[2] - zs[1]
    
    ε = zeros(Float64, Nz)
    
    # kinetic term
    @. ε += ħc^2/2mc²*τ
    
    # t₀ term
    @. ε += (3/8)*t₀*ρ^2
    
    # t₃ term
    @. ε += (1/16)*t₃*ρ^3
    
    # yukawa term
    qmax = length(cnvl_coeff) - 1
    temp = π*a*a*V₀
    for iz in 1:Nz, q in 0:qmax
        if iz+q ≤ Nz
            ε[iz] += temp*ρ[iz]*cnvl_coeff[1+q]*ρ[iz+q]
        end
        if iz-q ≥ 1
            ε[iz] += temp*ρ[iz]*cnvl_coeff[1+q]*ρ[iz-q]
        elseif 1-(iz-q) ≤ Nz
            ε[iz] += temp*ρ[iz]*cnvl_coeff[1+q]*ρ[1-(iz-q)]
        end
    end
    
    E = sum(ε)*2Δz
end

function calc_total_energy_functional(param, zs, spEs, Efermi, ρ, τ)
    @unpack mc², ħc, t₀, t₃, a, V₀, cnvl_coeff = param
    Nz = length(zs)
    Δz = zs[2] - zs[1]
    
    ε = zeros(Float64, Nz)
    
    # kinetic term
    @. ε += ħc^2/4mc²*τ
    
    # t₃ term
    @. ε += -(1/32)*t₃*ρ^3
    
    E = sum(ε)*2Δz
    
    nstates = length(spEs)
    for i in 1:nstates
        σᵢ = (2mc²/(π*ħc*ħc))*(Efermi-spEs[i])
        E += 0.5(σᵢ*spEs[i] + ħc^2/2mc² * π/2 * σᵢ^2)
    end
    return E 
end

function average_density!(ρ, τ, ρ_new, τ_new)
    @. ρ = (ρ + ρ_new)/2
    @. τ = (τ + τ_new)/2
    return
end

function HF_calc(param, zs; iter_max=10, show=true, rtol=1e-5)
    Etots = Float64[]
    
    ρ = initial_density(param, zs)
    τ = similar(zs)
    
    ρ_new = similar(zs)
    τ_new = similar(zs)
    
    ψs, spEs, Πs = solve_Hamiltonian(param, zs, ρ)
    ψs, spEs, Πs = sort_states(ψs, spEs, Πs)
    
    Efermi = calc_fermi_energy(param, spEs)
    calc_density!(ρ, τ, param, zs, ψs, spEs, Πs, Efermi)
    push!(Etots, calc_total_energy(param, zs, ρ, τ))
    
    for iter in 1:iter_max
        ψs, spEs, Πs = solve_Hamiltonian(param, zs, ρ)
        ψs, spEs, Πs = sort_states(ψs, spEs, Πs)
        
        Efermi = calc_fermi_energy(param, spEs)
        calc_density!(ρ_new, τ_new, param, zs, ψs, spEs, Πs, Efermi)
        average_density!(ρ, τ, ρ_new, τ_new)
        push!(Etots, calc_total_energy(param, zs, ρ, τ))
        
        if iter > 1 && abs((Etots[end] - Etots[end-1])/Etots[end]) < rtol
            converge = true
            println("iteration converged at iter = $iter")
            break
        end
    end
    
    if show
        p = plot()
        plot!(p, zs, ρ)
        plot!(p, zs, τ)
        display(p)

        p = plot(Etots)
        display(p)
        
        Etot = Etots[end]
        Etot_functional = calc_total_energy_functional(param, zs, spEs, Efermi, ρ, τ)
        @show Efermi Etot Etot_functional
        show_states(ψs, spEs, Πs)
    end
    
end
    



function calc_norm(zs, ψ)
    Δz = zs[2] - zs[1]
    sqrt(dot(ψ, ψ)*2Δz)
end

function calc_sp_energy(param, Hmat, ψ)
    @unpack ħc, mc² = param
    return dot(ψ, Hmat, ψ)/dot(ψ, ψ) * (ħc^2/2mc²)
end

function imaginary_time_evolution!(ψs, spEs, Πs, param, zs
        ;Δt=0.1, 
        iter_max=20, 
        rtol=1e-5, 
        show=true)

    Nz = length(zs)
    Δz = zs[2] - zs[1]
    nstates = size(ψs, 2)
    
    Etots = Float64[] # history of total energy

    ρ = similar(zs)
    τ = similar(zs)
    vpot = similar(zs)
    
    converge = false
    for iter in 1:iter_max
        Efermi = calc_fermi_energy(param, spEs)
        calc_density!(ρ, τ, param, zs, ψs, spEs, Πs, Efermi)
        push!(Etots, calc_total_energy(param, zs, ρ, τ))
        
        calc_potential!(vpot, param, zs, ρ) 
        
        for i in 1:nstates
            Hmat = make_Hamiltonian(zs, vpot, Πs[i])
            
            @views ψs[:,i] = (I - 0.5Δt*Hmat)*ψs[:,i]
            @views ψs[:,i] = (I + 0.5Δt*Hmat)\ψs[:,i]
            
            # gram schmidt orthogonalization
            for j in 1:i-1
                if Πs[i] ≢ Πs[j]
                    continue
                end
                
                @views ψs[:,i] .-= ψs[:,j].*(dot(ψs[:,j],ψs[:,i])*2Δz)
            end
            
            # normalization
            @views ψs[:,i] ./= calc_norm(zs, ψs[:,i])
            
            @views spEs[i] = calc_sp_energy(param, Hmat, ψs[:,i])
        end
        
        ψs, spEs, Πs = sort_states(ψs, spEs, Πs)
        
        if iter > 1 && abs((Etots[end] - Etots[end-1])/Etots[end]) < rtol
            converge = true
            if show
                println("\n iteration converged at iter = $iter \n")
            end
            break
        end
    end
    
    if !converge
        println("no convergence")
    end
    
    return ρ, τ, Etots
end



function HF_calc_with_imaginary_time_step(
        ;σ = 1.4,
        Δz=0.1, 
        Nz=100, 
        Δt=0.1, 
        iter_max=100, 
        show=true)

    param = PhysicalParam(σ=1.4)
    @unpack zs = param

    ρ₀ = initial_density(param, zs)
    ψs, spEs, Πs = solve_Hamiltonian(param, zs, ρ₀)
    ψs, spEs, Πs = sort_states(ψs, spEs, Πs)
    
    ρ, τ, Etots = imaginary_time_evolution!(ψs, spEs, Πs, param, zs
        ; Δt=Δt, 
        iter_max=iter_max, 
        show=show)
    
    Efermi = calc_fermi_energy(param, spEs)
    if show
        p = plot()
        plot!(p, zs, ρ₀; label="ρ₀")
        plot!(p, zs, ρ; label="ρ")
        plot!(p, zs, τ; label="τ")
        display(p)
        
        p = plot(Etots)
        display(p)
        
        Etot = Etots[end]
        Etot_functional = calc_total_energy_functional(param, zs, spEs, Efermi, ρ, τ)

        println("")
        @show Efermi Etot Etot_functional
        println("")

        println("single particle states: ")
        show_states(ψs, spEs, Πs)
    end

    return ψs, spEs, Πs, Efermi, ρ, τ
end
export HF_calc_with_imaginary_time_step




end # module
