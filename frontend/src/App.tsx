import React, { useState, useRef } from 'react'
import './App.css'

interface ExtractedData {
  label: string;
  text: string;
}

interface ApiResponse {
  success: boolean;
  data: ExtractedData[];
  annotated_image_base64: string | null;
  message: string;
}

function App() {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [results, setResults] = useState<ApiResponse | null>(null);
  const [dragActive, setDragActive] = useState<boolean>(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0]);
    }
  };

  const onButtonClick = () => {
    inputRef.current?.click();
  };

  const handleFile = async (selectedFile: File) => {
    if (!selectedFile.type.startsWith('image/')) {
      alert('Vui lòng chọn file hình ảnh!');
      return;
    }
    setFile(selectedFile);
    await uploadAndProcess(selectedFile);
  };

  const uploadAndProcess = async (selectedFile: File) => {
    setLoading(true);
    setResults(null);
    
    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await fetch('http://localhost:8000/api/v1/extract', {
        method: 'POST',
        body: formData,
      });
      
      const data: ApiResponse = await response.json();
      setResults(data);
    } catch (error) {
      console.error('Error uploading file:', error);
      alert('Lỗi kết nối tới Server API! Vui lòng đảm bảo Backend FastAPI đang chạy.');
    } finally {
      setLoading(false);
    }
  };

  const resetState = () => {
    setFile(null);
    setResults(null);
  };

  return (
    <div className="app-container">
      <header className="header">
        <h1>Invoice AI</h1>
        <p>Hệ thống nhận diện và trích xuất thông tin hóa đơn tự động</p>
      </header>

      <main className="main-content">
        <div className="glass-panel">
          {!file && !loading && (
            <div 
              className={`upload-zone ${dragActive ? "drag-active" : ""}`}
              onDragEnter={handleDrag}
              onDragLeave={handleDrag}
              onDragOver={handleDrag}
              onDrop={handleDrop}
              onClick={onButtonClick}
            >
              <div className="upload-icon">📄</div>
              <p className="upload-text">Kéo thả hóa đơn vào đây</p>
              <p className="upload-subtext">hoặc click để chọn file (JPG, PNG)</p>
              <input 
                ref={inputRef} 
                type="file" 
                className="file-input" 
                accept="image/*" 
                onChange={handleChange} 
              />
            </div>
          )}

          {loading && (
            <div className="upload-zone">
              <div className="spinner"></div>
              <p className="upload-text">AI đang phân tích...</p>
              <p className="upload-subtext">Quá trình này chỉ mất vài giây</p>
            </div>
          )}

          {file && !loading && results && (
            <div className="image-preview-container">
              {results.annotated_image_base64 ? (
                <img 
                  src={results.annotated_image_base64} 
                  alt="Annotated Invoice" 
                  className="image-preview"
                />
              ) : (
                <img 
                  src={URL.createObjectURL(file)} 
                  alt="Original Invoice" 
                  className="image-preview"
                />
              )}
            </div>
          )}
        </div>

        <div className="glass-panel">
          {results ? (
            <>
              <h2 style={{marginBottom: '1.5rem', fontSize: '1.5rem'}}>Kết quả trích xuất</h2>
              {results.success ? (
                <div className="results-grid">
                  {results.data.map((item, index) => (
                    <div key={index} className={`result-item ${item.label}`}>
                      <div className="result-label">{item.label}</div>
                      <div className="result-value">{item.text}</div>
                    </div>
                  ))}
                </div>
              ) : (
                <p style={{color: '#ef4444'}}>Lỗi trích xuất: {results.message}</p>
              )}
              <button className="btn-reset" onClick={resetState}>
                Xử lý hóa đơn khác
              </button>
            </>
          ) : (
             <div style={{height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-secondary)'}}>
               <p>Kết quả sẽ hiển thị ở đây</p>
             </div>
          )}
        </div>
      </main>
    </div>
  )
}

export default App
