module OneDimensionalStaticHF

export calc_cnvl_coeff, calc_potential!, HF_calc_with_imaginary_time_step

using Plots
using LinearAlgebra
using Parameters
using Arpack
using MyLibrary


"""
    struct PhysicalParam @deftype Float64

mc²: proton mass [MeV]

ħc: Planck constant ħ times speed of light c [MeV fm]

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
    σ = 1.4

    Δz = 0.1
    Nz::Int64 = 100

    zs::T = range(Δz/2, (Nz-1/2)*Δz, length=Nz)
    
    cnvl_coeff::Vector{Float64} = calc_cnvl_coeff(Δz, a)
end

@with_kw struct Densities 
    "ρ: number density"
    ρ::Vector{Float64}

    "τ: kinetic density"
    τ::Vector{Float64} = similar(ρ)
end


@with_kw struct SingleParticleStates 
    "nstates: number of single particle states"
    nstates::Int64

    "spEs: single particle energies [MeV]"
    spEs::Vector{Float64}; @assert length(spEs) === nstates

    "Πs: parities"
    Πs::Vector{Int64}; @assert length(Πs) === nstates 

    "ψs: wave functions"
    ψs::Matrix{Float64}; @assert size(ψs, 2) === nstates 

    "occ: occupation numbers"
    occ::Vector{Float64}; @assert length(occ) === nstates
end



"""
    calc_cnvl_coeff(param, zs; qmax=25)

Calculate convolution coefficients for Yukawa potential.
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



param = PhysicalParam(σ=1.4)
@unpack zs = param


function initial_density(param)
    @unpack a, ρ₀, σ, zs = param
    L = σ/ρ₀
    Δz = zs[2] - zs[1]
    
    ρ = similar(zs)
    @. ρ = ρ₀/(1 + exp((zs - 0.5L)/a))
    
    return Densities(ρ=ρ)
end

function test_initial_density(;σ=1.4)
    param = PhysicalParam(σ=σ)
    @time dens = initial_density(param)
    plot(zs, dens.ρ)
end



function calc_potential!(vpot, param, dens)
    @unpack mc², ħc, t₀, t₃, a, V₀, Nz, zs, cnvl_coeff = param
    @unpack ρ = dens
    
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

    dens = initial_density(param)
        
    vpot = similar(zs)
    @time calc_potential!(vpot, param, dens)
        
    plot(zs, vpot)
end



function make_Hamiltonian!(Hmat, param, vpot, Π)
    @assert Π == 1 || Π == -1
    @unpack Δz, Nz, zs = param
    @unpack dv, ev = Hmat
    
    @. dv = 2/Δz^2 + vpot
    dv[1] += (-1/Δz^2) * Π

    @. ev = -1/Δz^2
    
    return 
end


function test_make_Hamiltonian!(;σ=1.4, Π=1)
    @assert Π == 1 || Π == -1
    param = PhysicalParam(σ=σ)
    @unpack zs, Nz = param

    vpot = @. zs^2
    
    dv = zeros(Float64, Nz)
    ev = zeros(Float64, Nz-1)
    Hmat = SymTridiagonal(dv, ev)
    @time make_Hamiltonian!(Hmat, param, vpot, Π)
    
    vals, vecs = eigen(Hmat)
    vals[1:10] ./ 2
end



function initial_states!(vpot, Hmat, param, dens; Emax=0, nstates_max=10)
    @unpack Δz, Nz, zs, ħc, mc² = param
    
    ψs = zeros(Float64, Nz, nstates_max) # wave function
    spEs = zeros(Float64, nstates_max) # single particle energy
    Πs = zeros(Int64, nstates_max) 
    occ = zeros(Float64, nstates_max)
    
    istate = 0
    for Π in 1:-2:-1
        calc_potential!(vpot, param, dens)
        make_Hamiltonian!(Hmat, param, vpot, Π)
        
        #vals, vecs = eigs(Hmat; nev=nstates_max, which=:SM)
        vals, vecs = eigen(Hmat)
        
        # normalization
        @. vals *= ħc^2/2mc²
        @. vecs /= sqrt(2Δz)
        
        #N = length(vals[vals .< Emax])
        @views for i in 1:nstates_max 
            if vals[i] > Emax continue end
            istate += 1
            ψs[:,istate] = vecs[:,i]
            spEs[istate] = vals[i]
            Πs[istate] = Π
        end
    end

    states = SingleParticleStates(
        nstates=istate,
        spEs=spEs[1:istate],
        Πs=Πs[1:istate],
        ψs=ψs[:,1:istate],
        occ=occ[1:istate]
    )
    
    return states
end


function sort_states!(states)
    @unpack ψs, spEs, Πs = states
    p = sortperm(spEs)
    ψs[:] = ψs[:,p]
    spEs[:] = spEs[p]
    Πs[:] = Πs[p]
end

function calc_occ!(states, param, Efermi)
    @unpack mc², ħc = param 
    @unpack nstates, occ, spEs = states 

    for i in 1:nstates 
        if spEs[i] ≤ Efermi
            occ[i] = 2mc²/(π*ħc*ħc) * (Efermi - spEs[i])
        else
            occ[i] = 0 
        end
    end
end

function show_states(states)
    @unpack ψs, spEs, Πs, occ = states

    println("")
    for i in 1:size(ψs, 2)
        println("i = $i: ")
        @show spEs[i] Πs[i] occ[i]
        println("")
    end
end

function test_initial_states(;σ=1.4, Efermi=-20)
    param = PhysicalParam(σ=σ)
    @unpack zs, Nz = param

    dens = initial_density(param)

    vpot = similar(zs)

    dv = zeros(Float64, Nz)
    ev = zeros(Float64, Nz-1)
    Hmat = SymTridiagonal(dv, ev)

    @time states = initial_states!(vpot, Hmat, param, dens)
    @time sort_states!(states)
    @time calc_occ!(states, param, Efermi)
    
    @unpack nstates, ψs = states 
    p = plot()
    for i in 1:nstates
        plot!(p, zs, @views @. abs2(ψs[:,i]))
    end
    display(p)

    show_states(states)
    
    return
end





function func_fermi_energy(Efermi, param, states)
    @unpack mc², ħc, σ = param

    calc_occ!(states, param, Efermi)
    @unpack nstates, occ = states 
    
    f = 0.0
    for i in 1:nstates
        f += occ[i]
    end
    f -= σ
end

function calc_fermi_energy(param, states; ΔE=0.5)
    @unpack spEs = states
    Erange = -50:ΔE:0
    
    Efermi = 0.0
    found = false
    for i in 1:length(Erange)-1
        f₁ = func_fermi_energy(Erange[i], param, states)
        f₂ = func_fermi_energy(Erange[i+1], param, states)
        if f₁*f₂ < 0
            Efermi = bisect(func_fermi_energy, Erange[i], Erange[i+1], 
                args=(param, states))
            found = true
            break
        end
    end
    
    if !found
        error("fermi energy not found")
    end
    return Efermi
end

function test_calc_fermi_energy(σ=1.4)
    param = PhysicalParam(σ=σ)
    @unpack zs, Nz = param

    dens = initial_density(param)

    vpot = similar(zs)

    dv = zeros(Float64, Nz)
    ev = zeros(Float64, Nz-1)
    Hmat = SymTridiagonal(dv, ev)

    states = initial_states!(vpot, Hmat, param, dens)
    sort_states!(states)
    
    @time Efermi = calc_fermi_energy(param, states)
    @show Efermi func_fermi_energy(Efermi, param, states)
    show_states(states)
end
            



function first_deriv!(dψ, param, ψ, Π)
    @assert Π == 1 || Π == -1
    @unpack Nz, Δz = param
    
    #dψ[1] = (1-Π)*ψ[2]/2Δz
    dψ[1] = (ψ[2] - Π*ψ[1])/2Δz
    for iz in 2:Nz-1
        dψ[iz] = (ψ[iz+1] - ψ[iz-1])/2Δz
    end
    dψ[Nz] = -ψ[Nz-1]/2Δz
    
    return
end

function test_first_deriv!()
    param = PhysicalParam(σ=1.4)
    @unpack zs = param

    ψ = @. exp(-0.5zs*zs)
    
    dψ = similar(zs)
    first_deriv!(dψ, param, ψ, 1)
    
    dψ_exact = @. -zs*exp(-0.5zs*zs)
    
    plot(zs, dψ)
    plot!(zs, dψ_exact)
end



function calc_density!(dψ, dens, param, states)
    @unpack mc², ħc, zs = param
    @unpack ρ, τ = dens 
    @unpack nstates, ψs, spEs, Πs, occ = states
    
    fill!(ρ, 0)
    fill!(τ, 0)
    for i in 1:nstates
        @views ψ = ψs[:,i]
        first_deriv!(dψ, param, ψ, Πs[i])
        @. ρ += occ[i]*dot(ψ, ψ)
        @. τ += occ[i]*dot(dψ, dψ)
        @. τ += (π/2)*occ[i]^2*dot(ψ, ψ)
    end
end

function test_calc_density!()
    param = PhysicalParam(σ=1.4)
    @unpack zs, Nz, Δz = param

    dens = initial_density(param)
    p = plot()
    plot!(p, zs, dens.ρ; label="ρ₀")

    vpot = similar(zs)

    dv = zeros(Float64, Nz)
    ev = zeros(Float64, Nz-1)
    Hmat = SymTridiagonal(dv, ev)
    
    states = initial_states!(vpot, Hmat, param, dens)
    sort_states!(states)
    
    Efermi = calc_fermi_energy(param, states)
    calc_occ!(states, param, Efermi)
    
    dψ = similar(zs)
    @time calc_density!(dψ, dens, param, states)
    plot!(p, zs, dens.ρ; label="ρ")
    plot!(p, zs, dens.τ; label="τ")
    display(p)
    
    @show sum(dens.ρ)*2Δz
    
    return
end
    



function calc_total_energy(param, dens)
    @unpack mc², ħc, t₀, t₃, a, V₀, Nz, Δz, zs, cnvl_coeff = param
    @unpack ρ, τ = dens 
    
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

function calc_total_energy2(param, dens, states)
    @unpack mc², ħc, t₀, t₃, a, V₀, Nz, Δz, zs, cnvl_coeff = param
    @unpack ρ, τ = dens 
    @unpack nstates, spEs, occ = states
    
    ε = zeros(Float64, Nz)
    
    # kinetic term
    @. ε += ħc^2/4mc²*τ
    
    # t₃ term
    @. ε += -(1/32)*t₃*ρ^3
    
    E = sum(ε)*2Δz
    
    for i in 1:nstates
        E += 0.5(occ[i]*spEs[i] + ħc^2/2mc² * π/2 * occ[i]^2)
    end
    return E 
end




function calc_norm(param, ψ)
    @unpack Δz = param 
    sqrt(dot(ψ, ψ)*2Δz)
end

function calc_sp_energy(param, Hmat, ψ)
    @unpack ħc, mc² = param
    return dot(ψ, Hmat, ψ)/dot(ψ, ψ) * (ħc^2/2mc²)
end





function HF_calc_with_imaginary_time_step(;σ=1.4, Δz=0.1, Nz=100, Δt=0.1, iter_max=100, show_result=false)

    param = PhysicalParam(σ=σ, Δz=Δz, Nz=Nz)
    @unpack zs, Nz, Δz = param

    Etots = Float64[]

    dens = initial_density(param)

    vpot = zeros(Float64, Nz)
    dv   = zeros(Float64, Nz)
    ev   = zeros(Float64, Nz-1)
    dψ   = zeros(Float64, Nz)
    Hmat = SymTridiagonal(dv, ev)
    
    states = initial_states!(vpot, Hmat, param, dens)
    sort_states!(states)
    
    Efermi = calc_fermi_energy(param, states)
    calc_occ!(states, param, Efermi)

    calc_density!(dψ, dens, param, states)

    @unpack nstates, spEs, Πs, ψs, occ = states 

    converge = false
    for iter in 1:iter_max 
        calc_potential!(vpot, param, dens) 

        for i in 1:nstates 
            make_Hamiltonian!(Hmat, param, vpot, Πs[i])

            @views ψs[:,i] = (I - 0.5Δt*Hmat)*ψs[:,i]
            @views ψs[:,i] = (I + 0.5Δt*Hmat)\ψs[:,i]

            # gram schmidt orthogonalization
            for j in 1:i-1 
                if Πs[i] !== Πs[j] continue end 

                @views ψs[:,i] .-= ψs[:,j] .* (dot(ψs[:,j], ψs[:,i])*2Δz)
            end
            @views ψs[:,i] ./= calc_norm(param, ψs[:,i])
            @views spEs[i] = calc_sp_energy(param, Hmat, ψs[:,i])
        end

        sort_states!(states)
        Efermi = calc_fermi_energy(param, states)
        calc_occ!(states, param, Efermi)
        calc_density!(dψ, dens, param, states)

        push!(Etots, calc_total_energy(param, dens))
        if iter > 1 && isapprox(Etots[end], Etots[end-1], rtol=1e-5)
            converge = true
            if show_result
                println("iteration converged at iter = $(iter).")
            end
            break
        end
    end

    if !converge 
        error("no convergence in imaginary time step.")
    end

    if show_result
        println("")
        @show calc_fermi_energy(param, states)
        @show calc_total_energy(param, dens)
        @show calc_total_energy2(param, dens, states)
        show_states(states)

        p = plot()
        plot!(p, zs, dens.ρ; label="ρ")
        plot!(p, zs, dens.τ; label="τ")
        display(p)
    end

    return states, dens
end




function slab_mass_table(;σs=0.1:0.1:2.5, Δz=0.1, Nz=100, nstates_max=10)
    nσ = length(σs)
    Etots = zeros(Float64, nσ)
    Efermis = zeros(Float64, nσ)
    spEs = zeros(Float64, nσ, nstates_max)

    for iσ in 1:nσ 
        param = PhysicalParam(σ=σs[iσ], Δz=Δz, Nz=Nz)
        states, dens = HF_calc_with_imaginary_time_step(σ=σs[iσ], Δz=Δz, Nz=Nz)
        Etots[iσ] = calc_total_energy(param, dens)
        Efermis[iσ] = calc_fermi_energy(param, states)

        @unpack nstates = states 
        for istate in 1:nstates_max
            if istate <= nstates
                spEs[iσ,istate] = states.spEs[istate]
            else
                spEs[iσ,istate] = 0.0
            end
        end
    end

    p = plot(xlabel="Nucleons / fm²", ylabel="E [MeV]", ylim=(-50,-1))
    #plot!(p, σs, Etots; color=:black, linestyle=:dot, linewidth=2, label="E/Ω")
    plot!(p, σs, Efermis; color=:black, linestyle=:dash, label="ε_F")
    for istate in 1:nstates_max 
        if istate == 1
            plot!(p, σs, spEs[:,istate]; color=:black, label="e_n")
        else
            plot!(p, σs, spEs[:,istate]; color=:black, label=false)
        end
    end
    savefig("./1dimTDHF_figure/slab_mass_table_spEs")
    display(p)

    p2 = plot(xlabel="Nucleons / fm²", ylabel="E/Ω [MeV / fm²]")
    plot!(p2, σs, Etots; color=:black, st=:scatter, label="E/Ω", ylim=(-50,0))
    savefig("./1dimTDHF_figure/slab_mass_table_Etot")
    display(p2)

    return 
end


end # module
