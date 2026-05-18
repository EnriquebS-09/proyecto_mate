import streamlit as st
import sympy as sp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

# 1. Configuración de la interfaz
st.set_page_config(page_title="Calculadora de Optimización Completa", layout="centered")

# Definir la variable simbólica
x = sp.Symbol('x')

st.title("🧮 Herramienta de Optimización con Derivadas")
st.markdown("Análisis paso a paso, clasificación de puntos críticos y representación gráfica.")

# --- ENTRADA DE DATOS ---
input_usuario = st.text_input("Ingresa la función f(x):", "2*x**2 - 4*x - 1")

if st.button("Calcular y Graficar"):
    try:
        # Limpieza de entrada automática
        procesada = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', input_usuario)
        f = sp.sympify(procesada.replace('^', '**'))
        
        st.divider()
        
        # --- PASO 1: DERIVADAS ---
        st.subheader("1. Cálculo de Derivadas")
        f_prima = sp.diff(f, x)
        f_biprima = sp.diff(f_prima, x)
        
        st.write("Calculamos la primera derivada para hallar la pendiente y la segunda para la concavidad:")
        st.latex(f"f(x) = {sp.latex(f)}")
        st.latex(f"f'(x) = {sp.latex(f_prima)}")
        st.latex(f"f''(x) = {sp.latex(f_biprima)}")

        # --- PASO 2: PUNTOS CRÍTICOS (EL DESPEJE QUE QUERÍAS) ---
        st.subheader("2. Búsqueda de Puntos Críticos")
        st.write("Igualamos la primera derivada a cero ($f'(x) = 0$):")
        st.latex(f"0 = {sp.latex(f_prima)}")
        
        if f_prima.is_polynomial(x) and sp.degree(f_prima, x) == 1:
            termino_indep = f_prima.subs(x, 0)
            despeje_paso1 = f_prima - termino_indep 
            st.latex(f"{sp.latex(despeje_paso1)} = {sp.latex(-termino_indep)}")

        puntos_criticos = sp.solve(f_prima, x)
        for p in puntos_criticos:
            st.latex(f"x = {sp.latex(p)}")

        # --- PASO 3: CLASIFICACIÓN Y COORDENADAS ---
        st.subheader("3. Clasificación y Coordenadas")
        
        resultados = []
        for pc in puntos_criticos:
            if pc.is_real:
                valor_y = f.subs(x, pc)
                st.write(f"**Para el punto crítico $x = {pc}$:**")
                
                # Sustitución visual (Estilo image_785f3c.png)
                f_visual = f.subs(x, sp.Symbol(f'({pc})'))
                st.latex(f"f({pc}) = {sp.latex(f_visual)} = {sp.latex(valor_y)}")
                
                # Segunda derivada para clasificar
                segunda_der_eval = f_biprima.subs(x, pc)
                if segunda_der_eval > 0:
                    clase = "Mínimo"
                    nota = f"Como $f''({pc}) = {segunda_der_eval} > 0$, es un Mínimo."
                elif segunda_der_eval < 0:
                    clase = "Máximo"
                    nota = f"Como $f''({pc}) = {segunda_der_eval} < 0$, es un Máximo."
                else:
                    clase = "Inconcluso"
                    nota = "Segunda derivada igual a cero."
                
                st.info(nota)
                
                resultados.append({
                    "Punto Crítico (x)": float(pc),
                    "Valor en Y": float(valor_y),
                    "f''(x)": float(segunda_der_eval),
                    "Clasificación": clase
                })

        # --- TABLA DE RESULTADOS ---
        st.divider()
        st.subheader("📊 Tabla de Resultados Finales")
        if resultados:
            df = pd.DataFrame(resultados)
            st.table(df)

            # --- GRÁFICA (LA NUEVA SECCIÓN) ---
            st.subheader("📈 Representación Gráfica")
            
            # Crear los puntos para la curva
            pc_eje = resultados[0]["Punto Crítico (x)"]
            x_vals = np.linspace(pc_eje - 5, pc_eje + 5, 400)
            f_num = sp.lambdify(x, f, "numpy")
            y_vals = f_num(x_vals)

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(x_vals, y_vals, label=f"f(x) = {input_usuario}", color='#1f77b4', linewidth=2)
            
            # Marcar cada punto crítico con un punto rojo
            for res in resultados:
                ax.plot(res["Punto Crítico (x)"], res["Valor en Y"], 'ro')
                ax.annotate(f'{res["Clasificación"]}\n({res["Punto Crítico (x)"]}, {res["Valor en Y"]})', 
                            xy=(res["Punto Crítico (x)"], res["Valor en Y"]), 
                            xytext=(5, 5), textcoords='offset points')

            ax.axhline(0, color='black', lw=0.5)
            ax.axvline(0, color='black', lw=0.5)
            ax.grid(True, linestyle='--', alpha=0.6)
            st.pyplot(fig)
            
        else:
            st.warning("No se detectaron puntos críticos reales.")

    except Exception as e:
        st.error(f"Error: {e}. Asegúrate de usar el formato correcto (ej: 2*x**2-4*x-1)")