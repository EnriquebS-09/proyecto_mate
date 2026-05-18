import streamlit as st
import sympy as sp
import pandas as pd

# 1. Definir la variable matemática
x = sp.Symbol('x')

st.title("Calculadora de Máximos y Mínimos")
st.write("Herramienta para optimización usando la segunda derivada.")

# 2. INTERFAZ: Input del usuario
funcion_str = st.text_input("Ingresa la función f(x):", "-x**2 + 4*x + 5")

if st.button("Calcular"):
    try:
        # Convertir el texto a una expresión matemática
        # Nota: en Python se usa ** para potencias en lugar de ^
        funcion_str = funcion_str.replace('^', '**')
        f = sp.sympify(funcion_str)

        # 3. LÓGICA: Calcular derivadas
        f_prima = sp.diff(f, x)
        f_biprima = sp.diff(f_prima, x)

        # Encontrar las raíces de la primera derivada (puntos críticos)
        puntos_criticos = sp.solve(f_prima, x)

        resultados = []

        # 4. LÓGICA: Clasificación
        for pc in puntos_criticos:
            if pc.is_real: # Solo nos importan los números reales
                # Evaluar la segunda derivada en el punto crítico
                valor_segunda_der = f_biprima.subs(x, pc)
                
                # Encontrar la coordenada Y evaluando en la función original
                y_val = f.subs(x, pc)

                # Clasificar
                if valor_segunda_der > 0:
                    tipo = "Mínimo"
                elif valor_segunda_der < 0:
                    tipo = "Máximo"
                else:
                    tipo = "Indeterminado / Punto de silla"

                # Guardar en nuestra lista de resultados
                resultados.append({
                    "Punto Crítico (x)": float(pc), 
                    "Valor (y)": float(y_val), 
                    "Clasificación": tipo
                })

        # 5. INTERFAZ: Mostrar en una tabla
        if resultados:
            st.write("### Resultados")
            df = pd.DataFrame(resultados)
            st.table(df) # Esto renderiza la tabla que te piden
            
            st.write(f"**Primera derivada:** $f'(x) = {sp.latex(f_prima)}$")
            st.write(f"**Segunda derivada:** $f''(x) = {sp.latex(f_biprima)}$")
        else:
            st.warning("No se encontraron puntos críticos reales para esta función.")

    except Exception as e:
        st.error("Error al procesar la función. Asegúrate de escribirla correctamente (ej. usar * para multiplicar: 4*x).")