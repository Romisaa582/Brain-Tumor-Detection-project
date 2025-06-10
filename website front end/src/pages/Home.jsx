import React from 'react'
import Header from '../components/Header'
import SpecialityMenu from '../components/SpecialityMenu'
import Partners from '../components/Partners'
import TopDoctors from '../components/TopDoctors'
import Banner from '../components/Banner'

const Home = () => {
  return (
    
    <div>
        <Header/>
        <SpecialityMenu/>
        <TopDoctors/>
        <Banner/>
        <Partners />
    </div>
  )
}

export default Home