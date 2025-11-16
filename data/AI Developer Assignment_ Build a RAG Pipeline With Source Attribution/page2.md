```latex
\begin{itemize}
    \item \texttt{source\_filename}
    \item \texttt{page\_number} (if available)
    \item \texttt{chunk\_id}
    \item \texttt{embedding\_vector}
\end{itemize}

\section{3. Retrieval Module}
Given a query:
\begin{itemize}
    \item Convert to embedding
    \item Perform vector similarity search
    \item Return top relevant chunks
    \item Show metadata (file name + chunk numbers)
\end{itemize}

\section{4. LLM Answer Generation}
Use any LLM (OpenAI GPT, Llama, Mistral, etc.).

The answer must:
\begin{itemize}
    \item Be grounded only on retrieved chunks
    \item Contain citations like:
    \begin{itemize}
        \item \texttt{[Source: file.pdf | chunk 12]}
        \item OR
        \item \texttt{According to Document: product\_manual.pdf (Chunk \#4)}
    \end{itemize}
\end{itemize}

\section{5. Final Output Format (Mandatory)}
When user asks a question like:

\texttt{``What are the safety precautions for machine X?''}

Your system must return:
```